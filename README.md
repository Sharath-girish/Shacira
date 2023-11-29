# SHACIRA: Scalable HAsh-grid Compression for Implicit Neural Representations (ICCV 2023)

Official code for our [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Girish_SHACIRA_Scalable_HAsh-grid_Compression_for_Implicit_Neural_Representations_ICCV_2023_paper.pdf) <br>
**SHACIRA: Scalable HAsh-grid Compression for Implicit Neural Representations** <br>
Sharath Girish, Abhinav Shrivastava, Kamal Gupta <br>
University of Maryland, College Park <br>
_ICCV 2023_ <br>

This repository is built on top of an earlier version of [Kaolin Wisp](https://github.com/NVIDIAGameWorks/kaolin-wisp), a Pytorch library for working with neural fields.
For a full detailed overview of running the scripts and their functions and capabilities, we recommend checking out their repository.

## Setup

Clone repository

```shell
git clone git@github.com:Sharath-girish/shacira.git
cd shacira
```

Setup environment on conda 
```shell
conda create -n "env_shacira" python=3.9.5
conda activate env_shacira
```
or virtualenv
```shell
python -m venv env_shacira
source env_shacira/bin/activate
```

Install Pytorch packages along with Kaolin Wisp packages
```shell
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install kaolin==0.13.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.1_cu113.html
```

Build and install other dependencies as well as CUDA kernels for grid interpolation
```shell
pip install -r requirements.txt
pip install -r requirements_app.txt
python setup.py develop
```

## Usage

The repository contains scripts for training INRs for images and NeRFs under the `app` folder with their respective config files containing 
the default argparse arguments. 

### Images
Script for training INRs on images. Input is a directory containing set of images to train different INRs on independently.
Example run for the [Kodak dataset](https://r0k.us/graphics/kodak/).
```
python3 app/image/main_image.py --config app/image/configs/kodak.yaml --dataset-path /path/to/kodak/directory
```

### NeRFs

Example script for training compressible feature grids on a scene in the RTMV dataset available at the [dataset project page](http://www.cs.umd.edu/~mmeshry/projects/rtmv/).
```
python3 app/nerf/main_nerf.py --config app/nerf/configs/nerf_latents.yaml --multiview-dataset-format rtmv --mip 2 --bg-color white --raymarch-type voxel --num-steps 16 --num-rays-sampled-per-img 4096 --dataset-num-workers 4 --dataset-path /path/to/scene
```
For the lego scene from the [original NeRF dataset](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi).
```
python3 app/nerf/main_nerf.py --config app/nerf/configs/nerf_latents.yaml --dataset-path /path/to/lego
```

