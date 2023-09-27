# NeRF: Neural Radiance Fields for View Synthesis

## Overview

Refer to [Kaolin-Wisp](https://github.com/NVIDIAGameWorks/kaolin-wisp) for an overview of various commands for running NeRFs.

### Triplanar Grid

The triplanar grid uses a simple AABB acceleration structure for raymarching, and a pyramid of triplanes in multiple resolutions.

This is an extension of the triplane described in [Chan et al. 2021](https://nvlabs.github.io/eg3d/), with support for multi-level features.

## How to Run

### RGB Data

Synthetic objects are hosted on the [original NeRF author's Google Drive](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi).

Training your own captured scenes is supported by preprocessing with  [Instant NGP's colmap2nerf](https://github.com/NVlabs/instant-ngp/blob/master/docs/nerf_dataset_tips.md#preparing-new-nerf-datasets) script. 

```
python3 app/nerf/main_nerf.py --config app/nerf/configs/nerf_lego.yaml --dataset-path /path/to/lego
```

The code supports any "standard" NGP-format datasets that has been converted with the scripts from the 
`instant-ngp` library. We pass in the `--multiview-dataset-format` argument to specify the dataset type, which
in this case is different from the RTMV dataset type used for the other examples. 

The `--mip` argument controls the amount of downscaling that happens on the images when they get loaded. This is useful
for datasets with very high resolution images to prevent overload on system memory, but is usually not necessary for 
reasonably sized images like the fox dataset.


### RGBD Data

For datasets which contain depth data, Wisp optimizes by pre-pruning the sparse acceleration structure.
That allows faster convergence.

RTMV data is available at the [dataset project page](http://www.cs.umd.edu/~mmeshry/projects/rtmv/).

Example run command for the V8 scene from the RTMV dataset for SHACIRA (using LatentGrid format) is

```
python3 app/nerf/main_nerf.py --config app/nerf/configs/nerf_V8.yaml --dataset-path /path/to/lego
```

Command for Instant-NGP using HashGrid format is using nerf_hash.yaml config while VQAD uses nerf_codebook.yaml config.
### Memory considerations

* For faster multiprocess dataset loading, if your machine allows it try setting
`--dataset-num-workers 16`. To disable the multiprocessing, you can pass in `--dataset-num-workers -1`.
* The ``--num-steps`` arg allows for a tradeoff between speed and quality. Note that depending on `--raymarch-type`, the meaning of this argument may slightly change:
  * 'voxel' - intersects the rays with the cells. Then among the intersected cells, each cell
  is sampled `num_steps` times.
  * 'ray' - samples `num_steps` along each ray, and then filters out samples which fall outside of occupied
  cells.
* Other args such as `base_lod`, `num_lods`, `codebook_bitwidth` and the number of epochs may affect the output quality.


