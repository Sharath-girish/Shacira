/*
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
 */

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

namespace wisp {
typedef unsigned int uint;

__device__ int32_t 
hash_index2d(
    const int2 pos,
    const int32_t resolution,
    const int32_t codebook_size
){
    int32_t index = 0;

    constexpr uint32_t primes[2] = { 1u, 2654435761u };

    if (resolution < codebook_size && 
        resolution * resolution < codebook_size) {
        index = pos.x + 
                pos.y * resolution;
    } else {
        index = (pos.x * primes[0] ^
                 pos.y * primes[1]) % codebook_size;
    }
    return index;
}

__inline__ __device__ float 
clamp(float x, float a, float b)
{
    return max(a, min(b, x));
}

template<typename scalar_t>
__global__ void
hashgrid_interpolate2d_cuda_kernel(
    const int64_t num_coords,
    const int32_t codebook_size,
    const int64_t feature_dim,
    const int32_t resolution,
    const int32_t lod_idx,
    const int32_t num_lods,
    const float* __restrict__ coords,
    const scalar_t* __restrict__ codebook,
    const int32_t *codebook_first_idx,
    scalar_t* __restrict__ feats
){
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int64_t stride = blockDim.x*gridDim.x;

    codebook = codebook + codebook_first_idx[lod_idx] * feature_dim; 

    for (int64_t i=tidx; i<num_coords; i+=stride) { 
        
        float2 x = make_float2(clamp(resolution * (coords[i*2+0] * 0.5 + 0.5), 0, resolution-1-1e-5), 
                               clamp(resolution * (coords[i*2+1] * 0.5 + 0.5), 0, resolution-1-1e-5));
        int2 pos = make_int2(floor(x.x), floor(x.y));
        float2 x_ = make_float2(x.x - static_cast<float>(pos.x), 
                                x.y - static_cast<float>(pos.y));
        float2 _x = make_float2(1.0 - x_.x, 1.0 - x_.y);

        float c000 = _x.x * _x.y;
        float c001 = _x.x * x_.y;
        float c010 = x_.x * _x.y;
        float c011 = x_.x * x_.y;
        // float c100 = x_.x * _x.y;
        // float c101 = x_.x * _x.y;
        // float c110 = x_.x * x_.y;
        // float c111 = x_.x * x_.y;
        
        int32_t corner_idx[4];
#       pragma unroll
        for (int j=0; j<4; ++j) {
            int2 corner;
            corner.x = pos.x + ((j & 2) >> 1);
            corner.y = pos.y + ((j & 1) >> 0);
            corner_idx[j] = hash_index2d(corner, resolution, codebook_size);
        }
        
        for (uint64_t j=0; j<feature_dim; ++j) {
            float feat =
                static_cast<float>(codebook[corner_idx[0]*feature_dim+j]) * c000 + 
                static_cast<float>(codebook[corner_idx[1]*feature_dim+j]) * c001 + 
                static_cast<float>(codebook[corner_idx[2]*feature_dim+j]) * c010 + 
                static_cast<float>(codebook[corner_idx[3]*feature_dim+j]) * c011;
            feats[num_lods*i*feature_dim+feature_dim*lod_idx+j] = static_cast<scalar_t>(feat);
        }
    }
} 

void hashgrid_interpolate2d_cuda_impl(
    int64_t num_coords, 
    int32_t codebook_size,
    int64_t feature_dim,
    int32_t resolution,
    int32_t lod_idx,
    int32_t num_lods,
    at::Tensor coords,
    at::Tensor codebook,
    at::Tensor codebook_first_idx,
    at::Tensor feats){

    int num_threads = 512;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats.type(), "hashgrid_interpolate2d_cuda", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(feats));
        auto stream = at::cuda::getCurrentCUDAStream();
        hashgrid_interpolate2d_cuda_kernel<<<(num_coords + num_threads - 1) / num_threads, num_threads, 0, stream>>>(
            num_coords,
            codebook_size,
            feature_dim,
            resolution,
            lod_idx,
            num_lods,
            coords.data_ptr<float>(),
            codebook.data_ptr<scalar_t>(),
            codebook_first_idx.data_ptr<int32_t>(),
            feats.data_ptr<scalar_t>()
        );
    }));
}

template<typename scalar_t>
__global__ void
hashgrid_interpolate2d_backward_cuda_kernel(
    const int64_t num_coords,
    const int32_t codebook_size,
    const int64_t feature_dim,
    const int32_t resolution,
    const int32_t lod_idx,
    const int32_t num_lods,
    const bool require_grad_coords,
    const float* __restrict__ coords,
    const scalar_t* __restrict__ codebook,
    const int32_t *__restrict__ codebook_first_idx,
    const scalar_t* __restrict__ grad_output, // N, feature_dim*num_lods
    scalar_t* __restrict__ grad_codebook, // codebook_size, feature_dim
    float* __restrict__ grad_coords // N, 2
){
    uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
    int64_t stride = blockDim.x*gridDim.x;

    grad_codebook = grad_codebook + codebook_first_idx[lod_idx] * feature_dim;
    codebook = codebook + codebook_first_idx[lod_idx] * feature_dim; 

    for (int64_t i=tidx; i<num_coords; i+=stride) { 
        
        float2 x = make_float2(clamp(resolution * (coords[i*2+0] * 0.5 + 0.5), 0, resolution-1-1e-5), 
                               clamp(resolution * (coords[i*2+1] * 0.5 + 0.5), 0, resolution-1-1e-5));
        int2 pos = make_int2(floor(x.x), floor(x.y));
        float2 x_ = make_float2(x.x - static_cast<float>(pos.x), 
                                x.y - static_cast<float>(pos.y));
        float2 _x = make_float2(1.0 - x_.x, 1.0 - x_.y);


        float coeffs[4];
        coeffs[0] = _x.x * _x.y;
        coeffs[1] = _x.x * x_.y;
        coeffs[2] = x_.x * _x.y;
        coeffs[3] = x_.x * x_.y;
        // coeffs[4] = x_.x * _x.y;
        // coeffs[5] = x_.x * _x.y;
        // coeffs[6] = x_.x * x_.y;
        // coeffs[7] = x_.x * x_.y;
        
        int32_t corner_idx[4];

#       pragma unroll
        for (int j=0; j<4; ++j) {
            int2 corner;
            corner.x = pos.x + ((j & 2) >> 1);
            corner.y = pos.y + ((j & 1) >> 0);
            corner_idx[j] = hash_index2d(corner, resolution, codebook_size);
        }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
        if (std::is_same<scalar_t, at::Half>::value) {
            for (uint64_t j=0; j<feature_dim; j += 2) {
#           pragma unroll
                for (int k=0; k<4; ++k) {
                    uint64_t _idx = i*num_lods*feature_dim + lod_idx*feature_dim;
                    __half2 grad = reinterpret_cast<const __half2*>(grad_output)[(_idx + j) / 2];
                    grad = __floats2half2_rn(__half2float(grad.x) * coeffs[k],
                                             __half2float(grad.y) * coeffs[k]);
                    atomicAdd((__half2*)(grad_codebook + (corner_idx[k]*feature_dim + j)), grad);
                }
            }
        } else
#endif
        {
            for (uint64_t j=0; j<feature_dim; ++j) {
#           pragma unroll
                for (int k=0; k<4; ++k) {
                    float grad =
                        grad_output[i*num_lods*feature_dim + lod_idx*feature_dim + j] * coeffs[k];
                    atomicAdd((float*)(grad_codebook + (corner_idx[k]*feature_dim + j)), grad);
                }
            }
        }
        
        if (require_grad_coords) {
            for (uint64_t j=0; j<feature_dim; ++j) {
                float _grad_output = static_cast<float>(grad_output[i*num_lods*feature_dim+j]);

                grad_coords[i*2 + 0] += _grad_output * 
                    ((_x.y) * 
                    (static_cast<float>(codebook[corner_idx[2]*feature_dim+j]) -
                     static_cast<float>(codebook[corner_idx[0]*feature_dim+j])) +
                     (x_.y) * 
                    (static_cast<float>(codebook[corner_idx[3]*feature_dim+j]) -
                     static_cast<float>(codebook[corner_idx[1]*feature_dim+j])));
                
                grad_coords[i*2 + 1] += _grad_output * 
                    ((_x.x) * 
                    (static_cast<float>(codebook[corner_idx[1]*feature_dim+j]) -
                     static_cast<float>(codebook[corner_idx[0]*feature_dim+j])) +
                     (x_.x) * 
                    (static_cast<float>(codebook[corner_idx[3]*feature_dim+j]) -
                     static_cast<float>(codebook[corner_idx[2]*feature_dim+j])));
            }   
        }
    }
}

void hashgrid_interpolate2d_backward_cuda_impl(
    int64_t num_coords, 
    int32_t codebook_size,
    int64_t feature_dim,
    int32_t resolution,
    int32_t lod_idx,
    int32_t num_lods,
    bool require_grad_coords,
    at::Tensor coords,
    at::Tensor codebook,
    at::Tensor codebook_first_idx,
    at::Tensor grad_output,
    at::Tensor grad_codebook,
    at::Tensor grad_coords){

    int num_threads = 512;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.type(), "hashgrid_interpolate2d_backward_cuda", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(grad_codebook));
        auto stream = at::cuda::getCurrentCUDAStream();
        hashgrid_interpolate2d_backward_cuda_kernel<<<(num_coords + num_threads - 1) / num_threads, num_threads, 0, stream>>>(
            num_coords,
            codebook_size,
            feature_dim,
            resolution,
            lod_idx,
            num_lods,
            require_grad_coords,
            coords.data_ptr<float>(),
            codebook.data_ptr<scalar_t>(),
            codebook_first_idx.data_ptr<int32_t>(),
            grad_output.data_ptr<scalar_t>(),
            grad_codebook.data_ptr<scalar_t>(),
            grad_coords.data_ptr<float>()
        );
    }));
}

} // namespace wisp
