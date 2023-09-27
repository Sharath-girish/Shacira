# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import os
from typing import Callable, List
import logging as log
import torch
from math import prod
from PIL import Image
from wisp.datasets.base_datasets import BaseImageDataset
from wisp.datasets.batch import ImageBatch
import wisp.ops.mesh as mesh_ops
from wisp.ops.image import load_rgb, load_rgb_tensor
from wisp.ops.raygen import generate_2d_grid

_SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp",
                      ".JPG", ".JPEG")
""" Supported image formats this dataset can load """
    
class MultiImageDataset(BaseImageDataset):
    """ 
    A dataset class for images and coordinate loading. The dataset loads images in the directory 
    with the supported formats.
    """

    def __init__(self,
                 dataset_path: str,
                 dataset_num_workers: int,
                 split: str,
                 transform: Callable = None,
                 num_samples: int = -1,
                 sample_mode: str='full',
                 ):
        """Construct by loading the image and generating an initial dataset of coords.

        Args:
            dataset_path (str): Path to images.
            dataset_num_workers (int): The number of workers to spawn for multiprocessed loading.
                Included for compatibility
            split (str): Any of 'train', 'val'.
                Currently used for keeping track of the dataset purpose and not used internally.
            transform (Optional[Callable]):
                A transform function applied per batch when data is accessed with __get_item__.
                Not used for image coordinate batch loading.
            num_samples (int): Number of samples/coords per batch.
                -1 for all coords in the image.
            sample_mode (str): Different sample methods to apply for each batch of coords:
            pregenerated coordinate grid methods: (Full coordinate grid stored in memory for sampling)
                'full' - all coords will be sampled from the coordinate grid.
                'woreplace' - num_samples coords will be sampled randomly from the coordinate grid
                            without replacement.
                'sequential' - num_samples coords will be sampled sequentially from the coordinate grid
            on-the-fly coordinate grid methods:
                'wreplace' - num_samples coords will be sampled randomly from the coordinate grid
                            with replacement, more memory efficient for large images.
                'eval' - num_samples coords sampled in sequential order
        """
        super().__init__(dataset_path=dataset_path, 
                         dataset_num_workers=dataset_num_workers,
                         split=split, 
                         transform=transform)

        self.dataset_path = dataset_path

        self.image_list = []
        for imagename in sorted(os.listdir(self.dataset_path)):
            if any([imagename.endswith(ext) for ext in _SUPPORTED_FORMATS]):
                self.image_list.append(os.path.join(self.dataset_path,imagename))

        self.num_images = len(self.image_list)
        # Sampling args
        self.sample_mode = sample_mode
        # If sampling all coords at once, set num_samples to -1
        if self.sample_mode == 'full': 
            num_samples = -1
        self.num_samples = num_samples

        # Maintain index of image in folder
        self.image_idx = 0

    def create_split(self, split: str):
        """ Creates a dataset with the same parameters and a different split.
        This is a convenient way of creating validation datasets, while making sure they're compatible
        with the train dataset.

        All settings except for split and transform will be copied from the current dataset.

        Args:
            split (str): The dataset split to use, corresponding to the transform file to load.
                Options: 'train', 'val'
            transform (Optional[Callable]):
                Transform function applied per batch when data is accessed with __get_item__.
                For example: ray sampling, to filter the amount of rays returned per batch.
                When multiple transforms are needed, the transform callable may be a composition of multiple Callable.
        """
        return MultiImageDataset(
            dataset_path=self.dataset_path,
            dataset_num_workers=self.dataset_num_workers,
            split=split,
            transform=self.transform,
            num_samples=self.num_samples,
            sample_mode='eval'
        )
    
    def load_next(self):
        """Load next image in multi image dataset"""
        self.data = dict()
        self.image_path = self.image_list[self.image_idx]

        # Calls load_singleprocess or load_multiprocess
        self.load()

        # Controls behavior of trainer if coordinates are same across iterations
        # For faster training, no need to iterate over dataloader if data is static
        self.static_coords = ((self.num_samples == -1) or 
                              (self.num_samples >= prod(self.image_size))) 
        # Increment image index
        self.image_idx += 1
        
    def resample(self) -> None:
        """Resample coordinate grid if using woreplace sampling mode. No resampling
        necessary for other modes.
        """
        if self.sample_mode == 'woreplace':
            self.shuffle_idx = torch.randperm(self.data['orig_coords'].shape[0])
            self.data['coords'] = self.data['orig_coords'][self.shuffle_idx]
            self.data['rgb'] = self.data['orig_rgb'][self.shuffle_idx]

    def load_singleprocess(self) -> None:
        """Initializes the dataset by loading an image and sampling coordinates from it.
        This function uses the main process to load the dataset, without spawning any workers.
        """

        image = load_rgb_tensor(self.image_path)
        self.data['orig_rgb'] = image.reshape(3,-1).permute(1,0)
        C, H, W = image.shape
        self.image_size = (H,W)
        # Create the coordinate grid if not using on-the-fly sampling methods such as 'eval' and 'wreplace'
        # Will be slow and take a lot of memory for large images
        if (self.sample_mode!='eval' and self.sample_mode != 'wreplace') or \
            (self.num_samples == -1 or self.num_samples > prod(self.image_size)):
            grid_y, grid_x = generate_2d_grid(W,H)
            # Normalize grid to [-1,1]
            # grid_y, grid_x = ((grid_y+0.5)/H-0.5)*2, ((grid_x+0.5)/W-0.5)*2 
            grid_y, grid_x = ((grid_y)/H-0.5)*2, ((grid_x)/W-0.5)*2 
            # grid_y, grid_x = (grid_y/H, grid_x/W)
            self.data['orig_coords'] = torch.stack([grid_y.reshape(-1,1), grid_x.reshape(-1,1)], dim=-1).reshape(-1,2)
            # If not sequential, shuffle the coordinates and corresponding rgb values
            if self.sample_mode != 'sequential':
                self.shuffle_idx = torch.randperm(self.data['orig_coords'].shape[0])
                self.data['coords'] = self.data['orig_coords'][self.shuffle_idx]
                self.data['rgb'] = self.data['orig_rgb'][self.shuffle_idx]
            else:
                self.data['coords'] = self.data['orig_coords']
                self.data['rgb'] = self.data['orig_rgb']

    @classmethod
    def is_root_of_dataset(cls, root: str, files_list: List[str]) -> bool:
        """ Each dataset may implement a simple set of rules to distinguish it from other datasets.
        Rules should be unique for this dataset type, such that given a general root path, Wisp will know
        to associate it with this dataset class.

        Datasets which don't implement this function should be created explicitly.

        Args:
                root (str): A path to the root directory of the dataset.
                files_list (List[str]): List of files within the dataset root, without their prefix path.
        Returns:
                True if the root dir contains atleast one image with supported format
        """
        return any([any([filename.endswith(ext) for ext in _SUPPORTED_FORMATS]) \
                    for filename in files_list])

    def __len__(self):
        """Return length of dataset, as number of batches of samples for current image.
           NOT number of images in dataset."""
        if self.num_samples == -1 or self.num_samples > prod(self.image_size):
            num_batches = 1
        else:
            num_batches = prod(self.image_size) // self.num_samples
            num_batches += 1 if prod(self.image_size) % self.num_samples > 0 else 0
        return num_batches

    def __getitem__(self, idx) -> ImageBatch:
        """Retrieve a batch of sample coordinates and their rgb values.

        Returns:
            (ImageBatch): A batch of coordinates and their RGB values. The fields can be accessed as a dictionary:
                "coords" - a torch.Tensor of the 2d coordinates of each sample
                "rgb" - a torch.Tensor of the RGB pixel values at each coordinate.
        """
        if self.num_samples == -1 or self.num_samples > prod(self.image_size):
            # Single batch consisting of all coordinates of the image
            out = ImageBatch(
                coords=self.data["coords"],
                rgb=self.data["rgb"]
            )
        elif self.sample_mode == 'woreplace' or self.sample_mode == 'sequential':
            # Sample current batch of coordinates from pregenerated coordinate grid
            start_idx = idx * self.num_samples
            end_idx = min(start_idx + self.num_samples, prod(self.image_size))
            out = ImageBatch(
                coords=self.data["coords"][start_idx:end_idx],
                rgb=self.data["rgb"][start_idx:end_idx]
            )
        else:
            if self.sample_mode == 'eval':
                start_idx = idx * self.num_samples
                end_idx = min(start_idx + self.num_samples, prod(self.image_size))
                idx = torch.arange(start_idx, end_idx, dtype=torch.long).reshape(-1,1)
            elif self.sample_mode == 'wreplace':
                idx = torch.randint(0, prod(self.image_size), size=(self.num_samples,1))
            # H,W = self.image_size
            # coord_idx = torch.div(idx, torch.Tensor([W,1]).reshape(1,-1), rounding_mode='floor').long()
            # coord_idx = torch.remainder(coord_idx, torch.Tensor([H,W]).reshape(1,-1).to(coord_idx)).long()
            # # coord_idx = ((coord_idx+0.5)/torch.Tensor([H,W]).reshape(1,-1)-0.5)*2
            # coord_idx = ((coord_idx)/torch.Tensor([H,W]).reshape(1,-1).to(coord_idx)-0.5)*2
            # # coord_idx = coord_idx/torch.Tensor([H,W]).reshape(1,-1)

            out = ImageBatch(
                coords=idx,
                rgb=self.data["orig_rgb"][idx[:,0]]
            )
        return out
    
    def transform_coords(self, coord_idx, device):
        """Transforms the coordinates to the range [-1,1]"""
        if self.num_samples == -1 or self.num_samples > prod(self.image_size) or \
           self.sample_mode == 'woreplace' or self.sample_mode == 'sequential':
            return coord_idx
        else:
            H,W = self.image_size
            coord_idx = torch.div(coord_idx.to(device), torch.Tensor([W,1]).reshape(1,-1).to(device), rounding_mode='floor').long()
            coord_idx = torch.remainder(coord_idx, torch.Tensor([H,W]).reshape(1,-1).to(coord_idx)).long()
            coord_idx = ((coord_idx)/torch.Tensor([H,W]).reshape(1,-1).to(coord_idx)-0.5)*2
            return coord_idx

    
    @property
    def coordinates(self) -> torch.Tensor:
        """ Returns the coordinates of samples stored in this dataset. """
        return self.data.get("coords")
    
    @property
    def pixels(self) -> torch.Tensor:
        """ Returns the coordinates of samples stored in this dataset. """
        return self.data.get("rgb")