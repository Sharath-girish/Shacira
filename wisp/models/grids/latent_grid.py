# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from __future__ import annotations
from typing import Dict, Set, Any, Type, List
import os
import torch
import torch.nn as nn
import numpy as np
import kaolin.ops.spc as spc_ops
import wisp.ops.grid as grid_ops
from wisp.accelstructs import OctreeAS, BaseAS, ASRaymarchResults
from wisp.models.grids import BLASGrid
from wisp.models.latent_decoders import *
from wisp.models.prob_models import *


class LatentGrid(BLASGrid):
    """A feature grid where hashed feature pointers are stored as multi-LOD grid nodes,
    and actual feature contents are stored in a latent hash table.
    The latents are quantized and decoded into a hashtable as in Instant-NGP
    (see: Muller et al. 2022, Instant-NGP: https://nvlabs.github.io/instant-ngp/)

    The occupancy state (e.g. BLAS, Bottom Level Acceleration Structure) is tracked separately from the feature
    volume, and relies on heuristics such as pruning for keeping it aligned with the feature structure.
    """
    def __init__(self,
        feature_dim        : int,
        latent_dim         : int,
        resolutions        : List[int],
        multiscale_type    : str   = 'sum',
        resolution_dim     : int = 3,
        feature_std        : float = 0.0,
        feature_bias       : float = 0.0,
        codebook_bitwidth  : int   = 8,
        blas_level         : int = 7,
        init_grid          : str = 'normal',
        conf_latent_decoder: Dict[str, Any] = {},
        conf_entropy_reg   : Dict[str, Any] = {},
    ):
        """Builds a LatentGrid instance, including the feature structure and an underlying BLAS for fast queries.

        Args:
            feature_dim (int): The dimension of the features stored on the grid.
             resolutions (List[int]): A list of resolutions to be used for each feature grid lod of the hash structure.
                i.e. resolutions=[562, 777, 1483, 2048] means that at LOD0, a grid of 562x562x562 nodes will be used,
                where each node is a hashed pointer to the feature table
                (note that feature table size at level L >= resolution of level L).
            multiscale_type (str): The type of multiscale aggregation.
                'sum' - aggregates features from different LODs with summation.
                'cat' - aggregates features from different LODs with concatenation.
                Note that 'cat' will change the decoder input dimension to num_lods * feature_dim.
            resolution_dim (int): Grid resolution dimension; 3(for videos/NeRFs) or 2 (for images)
            feature_std (float): The features are initialized with a Gaussian distribution with the given
                standard deviation.
            feature_bias (float): The features are initialized with a Gaussian distribution with the given mean.
            init_grid (str): The type of initialization for the feature grid. Choices are 'normal' and 'uniform'.
            codebook_bitwidth (int): Codebook dictionary_size is set as 2**bitwidth
            blas_level (int): The level of the octree to be used as the BLAS (bottom level acceleration structure).
            conf_latent_decoder (dict): Hyperparameters for the latent decoder if enabled
            conf_entropy_reg (dict): Hyperparameters for the probability models for entropy regularization
        """
        # Occupancy Structure
        self.blas_level = blas_level
        blas = OctreeAS.make_dense(level=blas_level)
        super().__init__(blas)
        self.dense_points = spc_ops.unbatched_get_level_points(self.blas.points,
                                                               self.blas.pyramid,
                                                               self.blas_level).clone()
        self.num_cells = self.dense_points.shape[0]
        self.occupancy = torch.zeros(self.num_cells)

        # Feature Structure - dims
        self.feature_dim = feature_dim
        # Latent dimension; By default is set to 0, which means that the latent dimension is the same as the feature dimension.
        self.latent_dim = feature_dim if latent_dim == 0 else latent_dim
        self.multiscale_type = multiscale_type
        self.feature_std = feature_std
        self.feature_bias = feature_bias
        self.codebook_bitwidth = codebook_bitwidth

        # Feature Structure - setup grid LODs
        self.resolutions = resolutions
        self.num_lods = len(resolutions)
        self.active_lods = [x for x in range(self.num_lods)]
        self.max_lod = self.num_lods - 1

        self.codebook_size = 2 ** self.codebook_bitwidth
        self.register_buffer("codebook_lod_sizes", torch.zeros(self.num_lods, dtype=torch.int32))
        self.register_buffer("codebook_lod_first_idx", torch.zeros(self.num_lods, dtype=torch.int32))
        

        self.codebook = []

        offset = 0
        for lod, res in enumerate(resolutions):
            num_pts = res ** resolution_dim
            fts = torch.zeros(min(self.codebook_size, num_pts), self.latent_dim)
            if init_grid == 'uniform':
                fts += (torch.rand_like(fts)-0.5) * 2 * self.feature_std
            elif init_grid == 'normal':
                fts += torch.randn_like(fts) * self.feature_std
            self.codebook.append(fts)
            self.codebook_lod_sizes[lod] = fts.shape[0]
            self.codebook_lod_first_idx[lod] = offset
            offset += fts.shape[0]
        self.codebook = nn.Parameter(torch.cat(self.codebook, dim=0))
        
        self.latent_dec = self.setup_decoders(conf_latent_decoder)
        self.prob_model = None
        self.noise = None
        if conf_latent_decoder['ldecode_enabled'] and (conf_entropy_reg['entropy_reg']>0.0
            or conf_entropy_reg['entropy_reg_end']>0.0):
            self.prob_model = BitEstimator(self.latent_dim, num_layers=conf_entropy_reg['num_prob_layers'])
            self.noise_freq = conf_entropy_reg['noise_freq']

    def ent_loss(self, idx, is_val=False):
        if self.prob_model is None:
            return 0.0, 0.0
        else:
            noise = self.noise
            if self.noise_freq == 1:
               noise = torch.rand(self.codebook.shape).to(self.codebook)-0.5
            elif idx % self.noise_freq == 0:
               self.noise = torch.rand(self.codebook.shape).to(self.codebook)-0.5
               noise = self.noise
            weight = (self.codebook + noise) if not is_val else torch.round(self.codebook) 
            weight_p, weight_n = weight + 0.5, weight - 0.5
            prob = self.prob_model(weight_p) - self.prob_model(weight_n)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / np.log(2.0), 0, 50))
            return total_bits / self.codebook.shape[0], total_bits

    def size(self, use_torchac=False, use_prob_model=False):
        ldec_size = self.latent_dec.size(use_torchac)
        codebook_bits = 0
        for dim in range(self.codebook.size(1)):
            weight = torch.round(self.codebook[:,dim]).long()
            unique_vals, counts = torch.unique(weight, return_counts = True)
            if not use_prob_model:
                probs = counts/torch.sum(counts)
            else:
                assert self.prob_model is not None
                probs = self.prob_model(unique_vals+0.5,single_channel=dim) - self.prob_model(unique_vals-0.5,single_channel=dim)

            if not use_torchac:
                information_bits = torch.clamp(-1.0 * torch.log(probs + 1e-10) / np.log(2.0), 0, 1000)
                size_bits = torch.sum(information_bits*counts).item()
                codebook_bits += size_bits
            else:
                import torchac
                cdf = torch.cumsum(probs,dim=0)
                cdf = torch.cat((torch.Tensor([0.0]).to(cdf),cdf))
                cdf = cdf.unsqueeze(0).repeat(self.codebook.size(0),1)
                cdf = cdf/cdf[:,-1:] # Normalize the final cdf value just to keep torchac happy
                
                weight = weight - weight.min()
                unique_vals, counts = torch.unique(weight, return_counts = True)
                mapping = torch.zeros((weight.max().item()+1))
                mapping[unique_vals] = torch.arange(unique_vals.size(0)).to(mapping)
                weight = mapping[weight]
                cdf = torch.cumsum(counts/counts.sum(),dim=0)
                cdf = torch.cat((torch.Tensor([0.0]).to(cdf),cdf))
                cdf = cdf.unsqueeze(0).repeat(weight.size(0),1)
                cdf = cdf/cdf[:,-1:] # Normalize the final cdf value just to keep torchac happy
                byte_stream = torchac.encode_float_cdf(cdf.detach().cpu(), weight.detach().cpu().to(torch.int16), \
                                                        check_input_bounds=True)
                codebook_bits += len(byte_stream)*8

        return ldec_size, codebook_bits
        
    def setup_decoders(self, decoder_cfg):
        if not decoder_cfg['ldecode_enabled']:
            return DecoderIdentity()
        decoder_cfg['feature_dim'] = self.feature_dim
        decoder_cfg['latent_dim'] = self.latent_dim
        if decoder_cfg['ldecode_type'] == 'hierarchical':
            offsets = torch.cat((self.codebook_lod_first_idx, self.codebook_lod_sizes[-1:]))
            decoder = HierarchicalLatentDecoder(self.num_lods, offsets, decoder_cfg)
        elif decoder_cfg['ldecode_type'] == 'multi':
            decoder_cfg['num_entries'] = self.codebook.size(0)
            decoder = MultiLatentDecoder(**decoder_cfg)
            del decoder_cfg['num_entries']
        elif decoder_cfg['ldecode_type'] == 'single':
            decoder = LatentDecoder(**decoder_cfg)
        return decoder
    
    @classmethod
    def from_octree(cls,
                    feature_dim        : int,
                    latent_dim         : int   = 0,
                    base_lod           : int   = 2,
                    num_lods           : int   = 1,
                    multiscale_type    : str   = 'sum',
                    resolution_dim     : int = 3,
                    feature_std        : float = 0.0,
                    feature_bias       : float = 0.0,
                    codebook_bitwidth  : int   = 8,
                    blas_level         : int   = 7,
                    init_grid          : str = 'normal',
                    conf_latent_decoder: dict  = {},
                    conf_entropy_reg   : dict  = {}) -> LatentGrid:
        """
        Builds a hash grid using an octree sampling pattern.

        Args:
            feature_dim (int): The dimension of the features stored on the grid.
            base_lod (int): The base LOD of the feature grid.
                            This is the lowest LOD of for which features are defined.
            num_lods (int): The number of LODs for which features are defined. Starts at base_lod.
                            i.e. base_lod=4 and num_lods=5 means features are kept for levels 5, 6, 7, 8.
            multiscale_type (str): The type of multiscale aggregation.
                                   'sum' - aggregates features from different LODs with summation.
                                   'cat' - aggregates features from different LODs with concatenation.
                                   Note that 'cat' will change the decoder input dimension to num_lods * feature_dim.
            resolution_dim (int): Grid resolution dimension; 3(for videos/NeRFs) or 2 (for images)
            feature_std (float): The features are initialized with a Gaussian distribution with the given
                                 standard deviation.
            feature_bias (float): The features are initialized with a Gaussian distribution with the given mean.
            codebook_bitwidth (int): Codebook dictionary_size is set as 2**bitwidth
            blas_level (int): The level of the octree to be used as the BLAS (bottom level acceleration structure).
                The LatentGrid is backed
            init_grid (str): The type of initialization for the feature grid. Choices are 'normal' and 'uniform'.
            conf_latent_decoder (dict): Hyperparameters for the latent decoder if enabled
            conf_entropy_reg (dict): Hyperparameters for the probability models for entropy regularization
        """
        octree_lods = [base_lod + x for x in range(num_lods)]
        resolutions = [2 ** lod for lod in octree_lods]
        return cls(feature_dim=feature_dim, resolutions=resolutions, multiscale_type=multiscale_type,
                   feature_std=feature_std, feature_bias=feature_bias, codebook_bitwidth=codebook_bitwidth,
                   blas_level=blas_level, latent_dim=latent_dim, conf_latent_decoder=conf_latent_decoder,
                   conf_entropy_reg=conf_entropy_reg, resolution_dim=resolution_dim, init_grid=init_grid)

    @classmethod
    def from_geometric(cls,
                       feature_dim        : int,
                       num_lods           : int,
                       latent_dim         : int   = 0,
                       multiscale_type    : str   = 'sum',
                       resolution_dim     : int = 3,
                       feature_std        : float = 0.0,
                       feature_bias       : float = 0.0,
                       codebook_bitwidth  : int   = 8,
                       min_grid_res       : int   = 16,
                       max_grid_res       : int   = None,
                       blas_level         : int   = 7,
                       init_grid          : str = 'normal',
                       conf_latent_decoder: dict  = {},
                       conf_entropy_reg   : dict  = {}) -> LatentGrid:
        """
        Builds a hash grid using the geometric sequence initialization pattern from Muller et al. 2022 (Instant-NGP).
        This is an implementation of the geometric multiscale grid from
        instant-ngp (https://nvlabs.github.io/instant-ngp/).
        See Section 3 Equations 2 and 3 for more details.

        Args:
            feature_dim (int): The dimension of the features stored on the grid.
            num_lods (int): The number of LODs for which features are defined. Starts at lod=0.
                            i.e.  num_lods=16 means features are kept for levels 0, 1, 2, .., 14, 15.
            multiscale_type (str): The type of multiscale aggregation.
                                   'sum' - aggregates features from different LODs with summation.
                                   'cat' - aggregates features from different LODs with concatenation.
                                   Note that 'cat' will change the decoder input dimension to num_lods * feature_dim.
            feature_std (float): The features are initialized with a Gaussian distribution with the given
                                 standard deviation.
            resolution_dim (int): Grid resolution dimension; 3(for videos/NeRFs) or 2 (for images)
            feature_bias (float): The features are initialized with a Gaussian distribution with the given mean.
            codebook_bitwidth (int): Codebook dictionary_size is set as 2**bitwidth
            min_grid_res (int): min resolution of the feature grid.
            max_grid_res (int): max resolution of the feature grid.
            blas_level (int): The level of the octree to be used as the BLAS (bottom level acceleration structure).
            init_grid (str): The type of initialization for the feature grid. Choices are 'normal' and 'uniform'.
            conf_latent_decoder (dict): Hyperparameters for the latent decoder if enabled
            conf_entropy_reg (dict): Hyperparameters for the probability models for entropy regularization
        """
        b = np.exp((np.log(max_grid_res) - np.log(min_grid_res)) / (num_lods-1))
        resolutions = [int(1 + np.floor(min_grid_res*(b**l))) for l in range(num_lods)]
        return cls(feature_dim=feature_dim, resolutions=resolutions, multiscale_type=multiscale_type,
                   feature_std=feature_std, feature_bias=feature_bias, codebook_bitwidth=codebook_bitwidth,
                   blas_level=blas_level, latent_dim=latent_dim, conf_latent_decoder=conf_latent_decoder,
                   conf_entropy_reg=conf_entropy_reg, resolution_dim=resolution_dim, init_grid=init_grid)

    @classmethod
    def from_resolutions(cls,
                         feature_dim: int,
                         resolutions: List[int],
                         latent_dim: int = 0,
                         multiscale_type: str = 'sum',
                         resolution_dim: int = 3,
                         feature_std: float = 0.0,
                         feature_bias: float = 0.0,
                         codebook_bitwidth: int = 8,
                         blas_level: int = 7,
                         init_grid: str = 'normal',
                         conf_latent_decoder: dict = {},
                         conf_entropy_reg: dict = {}) -> LatentGrid:
        """
        Builds a hash grid from a list of resolution sizes (each entry contains a RES for the RES x RES x RES
        lod of nodes pointing at the actual hash table) .

        Args:
            feature_dim (int): The dimension of the features stored on the grid.
            resolutions (List[int]): A list of resolutions to be used for each feature grid lod of the hash structure.
                i.e. resolutions=[562, 777, 1483, 2048] means that at LOD0, a grid of 562x562x562 nodes will be used,
                where each node is a hashed pointer to the feature table
                (note that feature table size at level L >= resolution of level L).
            multiscale_type (str): The type of multiscale aggregation.
                                   'sum' - aggregates features from different LODs with summation.
                                   'cat' - aggregates features from different LODs with concatenation.
                                   Note that 'cat' will change the decoder input dimension to num_lods * feature_dim.
            resolution_dim (int): Grid resolution dimension; 3(for videos/NeRFs) or 2 (for images)
            feature_std (float): The features are initialized with a Gaussian distribution with the given
                                 standard deviation.
            feature_bias (float): The features are initialized with a Gaussian distribution with the given mean.
            codebook_bitwidth (int): Codebook dictionary_size is set as 2**bitwidth
            blas_level (int): The level of the octree to be used as the BLAS (bottom level acceleration structure)
            init_grid (str): The type of initialization for the feature grid. Choices are 'normal' and 'uniform'.
            conf_latent_decoder (dict): Hyperparameters for the latent decoder if enabled
            conf_entropy_reg (dict): Hyperparameters for the probability models for entropy regularization
        """
        return cls(feature_dim=feature_dim, resolutions=resolutions, multiscale_type=multiscale_type,
                   feature_std=feature_std, feature_bias=feature_bias, codebook_bitwidth=codebook_bitwidth,
                   blas_level=blas_level, latent_dim=latent_dim, conf_latent_decoder=conf_latent_decoder,
                   conf_entropy_reg=conf_entropy_reg, resolution_dim=resolution_dim, init_grid=init_grid)

    def freeze(self):
        """Freezes the feature grid.
        """
        self.codebook.requires_grad_(False)
        for p in self.latent_dec.parameters():
            p.requires_grad_(False)
        if self.prob_model is not None:
            for p in self.prob_model.parameters():
                p.requires_grad_(False)

    def interpolate(self, coords, lod_idx):
        """Query multiscale features.

        Args:
            coords (torch.FloatTensor): coords of shape [batch, num_samples, 3] or [batch, 3]
                For some grid implementations, specifying num_samples may allow for slightly faster trilinear
                interpolation. LatentGrid doesn't use this optimization, but allows this input type for compatability.
            lod_idx  (int): int specifying the index to ``active_lods``

        Returns:
            (torch.FloatTensor): interpolated features of shape
             [batch, num_samples, feature_dim] or [batch, feature_dim]
        """
        # Remember desired output shape
        output_shape = coords.shape[:-1]
        if coords.ndim == 3:    # flatten num_samples dim with batch for cuda call
            batch, num_samples, coords_dim = coords.shape  # batch x num_samples
            coords = coords.reshape(batch * num_samples, coords_dim)

        codebook = self.latent_dec(self.codebook)
        rep = False
        if codebook.size(1)==1:
            codebook = codebook.repeat(1,2)
            rep = True

        coords_dim = coords.shape[-1]
        hashgrid_fn = grid_ops.hashgrid2d if coords_dim == 2 else grid_ops.hashgrid
        
        feats = hashgrid_fn(coords, self.resolutions, self.codebook_bitwidth, lod_idx, codebook, self.codebook_lod_sizes, self.codebook_lod_first_idx)

        feats = feats[:,::2] if rep else feats

        if "RENDERING_FINAL" in os.environ:
            mask = torch.zeros_like(feats)
            mask[:,:lod_idx*self.feature_dim] = 1
            feats = feats * mask

        if self.multiscale_type == 'cat':
            return feats.reshape(*output_shape, feats.shape[-1])
        elif self.multiscale_type == 'sum':
            return feats.reshape(*output_shape, len(self.resolutions), feats.shape[-1] // len(self.resolutions)).sum(-2)
        else:
            raise NotImplementedError

    def raymarch(self, rays, raymarch_type, num_samples, level=None) -> ASRaymarchResults:
        """Mostly a wrapper over OctreeAS.raymarch. See corresponding function for more details.

        Important detail: the OctreeGrid raymarch samples over the coarsest LOD where features are available.
        """
        return self.blas.raymarch(rays, raymarch_type=raymarch_type, num_samples=num_samples, level=self.blas_level)

    def supported_blas(self) -> Set[Type[BaseAS]]:
        """ Returns a set of bottom-level acceleration structures this grid type supports """
        return {OctreeAS}

    def name(self) -> str:
        return "Latent Grid"

    def public_properties(self) -> Dict[str, Any]:
        """ Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        parent_properties = super().public_properties()
        active_lods = None if self.active_lods is None or len(self.active_lods) == 0 else \
            f'{min(self.active_lods)} - {max(self.active_lods)}'
        properties = {
            "Feature Dims": self.feature_dim,
            "Latent Dims": self.latent_dim,
            "Total LODs": self.max_lod,
            "Active feature LODs": active_lods,
            "Interpolation": 'linear',
            "Multiscale aggregation": self.multiscale_type,
            "HashTable Size": f"2^{self.codebook_bitwidth}"
        }
        return {**parent_properties, **properties}
