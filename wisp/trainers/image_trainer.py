# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import logging as log
import numpy as np
import os
import cv2
import copy
import torch
import random
import wandb
import time
import json
import math
from tqdm import tqdm
from dataclasses import dataclass, field
from torch.utils.tensorboard import SummaryWriter
from wisp.trainers import BaseTrainer, log_metric_to_wandb, log_images_to_wandb
from wisp.datasets import BaseImageDataset,ImageBatch, default_collate
from wisp.ops.image import hwc_to_chw, psnr, clamped_psnr, clamped_mse
from wisp.utils.schedulers import DecayScheduler
from wisp.models.grids import LatentGrid, HashGrid, CodebookOctreeGrid
from wisp.models.latent_decoders import LatentDecoder, MultiLatentDecoder

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
def print_kwargs(**kwargs):
    print(kwargs)

class ImageTrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ldec_lr_sched = DecayScheduler(self.extra_args["ldec_lr_warmup"], "fix", self.extra_args['ldec_lr'])
        
        # Update optimization state about the current train set and val set used
        self.scene_state.data.train_data.append(self.train_dataset)

        # self.log_fname = f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        self.log_dir = os.path.join(self.log_dir, self.exp_name)
        self.log_dir_cur = os.path.join(
            self.log_dir,
            os.path.splitext(os.path.basename(self.train_dataset.image_path))[0]
            # self.log_fname
        )
        os.makedirs(self.log_dir_cur,exist_ok=True)
        if os.path.exists(os.path.join(self.log_dir,'complete')):
            print(f'Experiment already exists at {self.log_dir}, exiting...')
            exit(1)

        log.info(f'Logging to {self.log_dir_cur}')
 

    def populate_scenegraph(self):
        """ Disable scenegraph update for images
        """
        pass

    def init_renderer(self):
        """Disable renderer for images
        """
        pass

    def init_dataloader(self):
        assert self.batch_size==1, "Image loader should be with batch size 1 only"
        self.train_data_loader = torch.utils.data.DataLoader(self.train_dataset,
                                            batch_size=self.batch_size,
                                            shuffle=False, pin_memory=False,
                                            num_workers=self.extra_args['dataloader_num_workers'])
        self.iterations_per_epoch = len(self.train_data_loader)

    def init_log_dict(self):
        """Custom logging dictionary.
        """
        super().init_log_dict()
        self.log_dict['rgb_loss'] = 0.0
        self.log_dict['total_loss'] = 0.0
        self.log_dict['PSNR'] = 0.0
        if self.extra_args["ldecode_enabled"] and self.entropy_reg_lambda > 0:
            self.log_dict['ent_loss'] = 0.0
            self.log_dict['net_kbytes_codebook'] = 0.0

    #######################
    # Data load
    #######################

    def reset_data_iterator(self):
        """Rewind the iterator for the new epoch if coordinates not static.
        """
        if not (self.train_dataset.static_coords and not self.is_first_iteration()):
            self.scene_state.optimization.iterations_per_epoch = len(self.train_data_loader)
            self.train_data_loader_iter = iter(self.train_data_loader)

    #######################
    # Training Events
    #######################
        
    def pre_training(self):
        """
        Override this function to change the logic which runs before the first training iteration.
        This function runs once before training starts.
        """
        self.best_state = {'rgb_loss':np.inf, 'PSNR': 0.0}
        if self.train_dataset.static_coords:
            #  Pointer to hold static data when initialized
            self.data = None

    def pre_epoch(self):
        """
        This function runs once before the epoch.
        """

        # Data resampling if not static every epoch
        if self.extra_args["resample"] and self.epoch % self.extra_args["resample_every"] == 0 \
            and self.epoch > 1 and not self.train_dataset.static_coords:
            self.resample_dataset()

        # If using latent decoder
        if self.extra_args["ldecode_enabled"]:
            grid = self.pipeline.nef.grid
            assert isinstance(grid, LatentGrid)

            # Entropy regularization loss schedule over training
            self.entropy_reg_lambda = self.entropy_reg_sched(self.epoch)

            # Temperature schedule for SGA while training
            if isinstance(grid.latent_dec, MultiLatentDecoder) or self.extra_args["use_sga"]:
                grid.latent_dec.temperature = self.temperature_sched(self.epoch)

            # Disable SGA after the decay period. Uses simple Straight Through Estimator after
            if self.extra_args["use_sga"] and (self.epoch)/self.max_epochs>self.extra_args['decay_period']:
                grid.latent_dec.use_sga = False
            
            # When straight through enabled, use only single decoder from MultiLatentDecoder
            if isinstance(grid.latent_dec, MultiLatentDecoder):
                grid.latent_dec.straight_through = (self.epoch/self.max_epochs)>self.extra_args['decay_period']
        
        # For static coordinates, pipeline is set to train only in the first epoch/iteration
        if not (self.train_dataset.static_coords and not self.is_first_iteration()):
            self.pipeline.train()

    def post_epoch(self):
        """
        Runs once after the epoch
        """

        if not self.train_dataset.static_coords:
            self.p_bar.close()

        self.log_dict['PSNR'] = self.log_dict['PSNR'] / self.iterations_per_epoch
        if isinstance(self.pipeline.nef.grid, LatentGrid) or \
            isinstance(self.pipeline.nef.grid, HashGrid) or \
            isinstance(self.pipeline.nef.grid, CodebookOctreeGrid):
            grid = self.pipeline.nef.grid
            use_torchac = self.total_iterations % self.extra_args["log_every"] == 0
            size_ldec, size_latents = grid.size(use_torchac=use_torchac, use_prob_model=False)
            size_remainder = sum([p.numel()*torch.finfo(p.dtype).bits for n,p in self.pipeline.named_parameters() if 'grid' not in n])
            if size_ldec>0:
                self.log_dict['ldec_size'] = size_ldec/8e3
            self.log_dict['latent_size'] = size_latents/8e3
            self.log_dict['remainder_size'] = size_remainder/8e3
            self.log_dict['total_size'] = (size_latents+size_remainder+size_ldec)/8e3
            self.log_dict['BPP'] = (size_latents+size_remainder+size_ldec)/math.prod(self.train_dataset.image_size)
            if isinstance(self.pipeline.nef.grid, LatentGrid):
                self.log_dict['rounding_loss'] = torch.mean(torch.abs(grid.codebook - torch.round(grid.codebook))).item()


        if self.train_dataset.static_coords and self.log_dict['rgb_loss'] < self.best_state['rgb_loss']:
            self.best_state['PSNR'] = self.log_dict['PSNR']
            self.best_state['BPP'] = self.log_dict['BPP']
            self.best_state['total_size'] = self.log_dict['total_size']
            self.best_state['rgb_loss'] = self.log_dict['rgb_loss']
            self.best_state['state_dict'] = copy.deepcopy(self.pipeline.state_dict())

        for k,v in self.log_dict.items():
            if 'loss' not in k:
                self.scene_state.optimization.metrics[k].append(v)
            else:
                self.scene_state.optimization.losses[k].append(v)
            
        if self.extra_args["log_every"] > -1 and self.epoch % self.extra_args["log_every"] == 0:
            self.log_cli()
            self.log_tb()

        # Render visualizations to tensorboard
        if self.render_tb_every > -1 and self.epoch % self.render_tb_every == 0 and not self.metrics_only:
            self.render_tb()


    def begin_epoch(self):
        """Begin epoch.
        """
        self.reset_data_iterator()
        self.pre_epoch()
        self.init_log_dict()
        self.epoch_start_time = time.time()
        if not self.train_dataset.static_coords:
            self.p_bar = tqdm(total=len(self.train_data_loader))

    def end_epoch(self):
        """End epoch.
        """
        current_time = time.time()
        elapsed_time = current_time - self.epoch_start_time
        self.epoch_start_time = current_time
        self.writer.add_scalar(f'time/elapsed_per_epoch', elapsed_time, self.epoch)
        if self.using_wandb:
            log_metric_to_wandb(f'time/elapsed_per_epoch', elapsed_time, self.epoch)

        self.post_epoch()

        # Save every N epochs to not overload the disk
        if self.extra_args["resume"] and self.epoch % self.extra_args["save_every"] == 0:
            self.save_state()
            
        # Validate every N epochs if enabled and coords not static (as train same as val then)
        if  self.extra_args["valid_every"] > -1 and \
            self.epoch % self.extra_args["valid_every"] == 0 and \
            self.epoch != 0 and not self.train_dataset.static_coords:
            with torch.no_grad():
                self.validate()

        if self.epoch < self.max_epochs:
            self.iteration = 0
            self.epoch += 1
        else:
            self.is_optimization_running = False

    def iterate_static(self):
        """Advances the training by one training step (batch) for static coordinates (one iteration)
        """
        assert self.iterations_per_epoch == 1
        if self.is_optimization_running:
            self.iteration += 1
            if self.is_first_iteration():
                self.pre_training()
            
            iter_start_time = time.time()
            self.begin_epoch()

            # Load data only first time since data is static
            if self.is_first_iteration():
                assert self.data is None
                self.data = self.next_batch()
            data = self.data
        
            # if self.is_any_iterations_remaining():
            self.pre_step()
            with torch.cuda.amp.autocast(self.enable_amp):
                self.step(data)
            self.post_step()
            iter_end_time = time.time()
            self.end_epoch()

            if not self.is_any_iterations_remaining():
                if self.using_wandb and (not self.is_optimization_running):
                    wandb.run.summary[f"image{self.train_dataset.image_idx}/train_time"] = (self.scene_state.optimization.elapsed_time+
                                                       (iter_end_time-iter_start_time))
                if not self.is_optimization_running:
                    self.post_training()
            self.scene_state.optimization.elapsed_time += iter_end_time - iter_start_time

    @torch.cuda.nvtx.range("ImageTrainer.step")
    def step(self, data):
        """Implement the optimization over image-space loss.
        """
        # Move to GPU only the first time if coords static
        if not (self.train_dataset.static_coords and not self.is_first_iteration()):
            data['coords'] = data['coords'].to(self.device)
            data['rgb'] = data['rgb'].to(self.device)
        coords = data['coords'].squeeze(0)
        coords = self.train_dataset.transform_coords(coords, self.device)
        img_gts = data['rgb'].squeeze(0)
        
        self.optimizer.zero_grad(set_to_none=True)
            
        loss = rgb_loss = 0

        with torch.no_grad():
            if (isinstance(self.pipeline.nef.grid, LatentGrid) and 
                self.extra_args["ldecode_enabled"] and 
                isinstance(self.pipeline.nef.grid.latent_dec, LatentDecoder) and
                self.extra_args["norm"]!='none' and 
                self.extra_args["norm_every"]%self.total_iterations==0):
                decoder = self.pipeline.nef.grid.latent_dec
                latents = self.pipeline.nef.grid.codebook
                if self.extra_args["norm"] == "max":
                    decoder.div.data = torch.max(torch.abs(latents.min(dim=0,keepdim=False)[0]),\
                                            torch.abs(latents.max(dim=0,keepdim=False)[0]))
                elif self.extra_args["norm"] == "std":
                    decoder.div.data = latents.std(dim=0,keepdim=False)

        pred = self.pipeline.nef(coords=coords, channels=["rgb"])[0]
        res = 1.0
        rgb_loss += ((pred - res * img_gts)**2).mean()

        self.log_dict['PSNR'] += clamped_psnr(pred, img_gts)
        # If coordinates static (single step per epoch), train is same as validation data
        # Best state is stored at end of validation otherwise
        if self.train_dataset.static_coords and rgb_loss < self.best_state['rgb_loss'] and not self.metrics_only:
            H,W = self.train_dataset.image_size
            sorted_image = pred.detach().cpu()[torch.argsort(self.train_dataset.shuffle_idx)]
            pred_image = torch.clamp(sorted_image.reshape(H,W,3)*255,0.,255.)\
                                     .numpy().astype(np.uint8)
            self.best_state['pred'] = pred_image

        loss += self.extra_args["rgb_loss"] * rgb_loss
        if self.extra_args["ldecode_enabled"] and self.entropy_reg_lambda > 0:
            avg_bits, net_bits = self.pipeline.nef.grid.ent_loss(
                                    self.total_iterations-1, 
                                    is_val=not self.pipeline.training
                                    )
            ent_loss = self.entropy_reg_lambda * avg_bits
            loss += ent_loss
            self.log_dict['ent_loss'] = ent_loss.item()
            self.log_dict['net_kbytes_codebook'] = net_bits/8/1024

        self.log_dict['rgb_loss'] += rgb_loss.item()
        self.log_dict['total_loss'] += loss.item()
        
        if (isinstance(self.pipeline.nef.grid, LatentGrid) and 
            self.extra_args["ldecode_enabled"] and 
            isinstance(self.pipeline.nef.grid.latent_dec, LatentDecoder) and
            self.extra_args["scale_grid_lr"]!='none'):
            with torch.no_grad():
                for group in self.optimizer.param_groups:
                    if group['name'] == 'grid':
                        norm = self.pipeline.nef.grid.latent_dec.scale_norm().item()
                        if self.extra_args["scale_grid_lr"]=='mul':
                            group['lr'] = self.extra_args["grid_lr"]* norm
                        elif self.extra_args["scale_grid_lr"]=='div':
                            group['lr'] = self.extra_args["grid_lr"]/ norm
                        elif self.extra_args["scale_grid_lr"]=='none':
                            group['lr'] = self.extra_args["grid_lr"]
                        else:
                            raise Exception(f"Unknown grid_lr scaling mode {self.extra_args['scale_grid_lr']}")
                        if self.total_iterations %100 == 0:
                            print("Latent learning rate: ", group['lr'], norm)
                    elif group['name'] == 'latent_dec':
                        group['lr'] = self.ldec_lr_sched(self.epoch)
                        # if self.total_iterations %100 == 0:
                        #     print("Latent decoder learning rate: ", group['lr'])

        with torch.cuda.nvtx.range("ImageTrainer.backward"):
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

    def post_step(self):
        if not self.train_dataset.static_coords:
            self.p_bar.update(1)

    def train(self):
        """
        Override this if some very specific training procedure is needed.
        """
        with torch.autograd.profiler.emit_nvtx(enabled=self.extra_args["profile"]):
            self.is_optimization_running = True
            while self.is_optimization_running:
                if self.train_dataset.static_coords:
                    self.iterate_static()
                else:
                    self.iterate()

    def validate(self):
        """Implement validation. 
        """
        val_dict = {'MSE':[], 'num_samples': [], 'clamped_MSE': [], 'PSNR': 0.0}
        val_dataset = self.train_dataset
        sample_mode_state = self.train_dataset.sample_mode
        # val_dataset.sample_mode = 'eval'
        # val_dataset.load()
        
        image_idx = val_dataset.image_idx
        val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=self.batch_size,
                                            shuffle=False, pin_memory=False,
                                            num_workers=self.extra_args['dataloader_num_workers'])
        iterations_per_epoch = len(val_data_loader)

        # For validation, disable SGA annealing if enabled
        if self.extra_args["ldecode_enabled"]:
            grid = self.pipeline.nef.grid
            assert isinstance(grid, LatentGrid)
            if self.extra_args["use_sga"]:
                use_sga_state = grid.latent_dec.use_sga
                grid.latent_dec.use_sga = False
            if isinstance(grid.latent_dec, MultiLatentDecoder):
                straight_through_state = grid.latent_dec.straight_through
                grid.latent_dec.straight_through = True

        self.pipeline.eval()
        H,W = self.train_dataset.image_size
        pred_image = torch.zeros(H*W,3)
        coord_idx = 0
        log.info('Starting validation for image {}/{}'.format(self.train_dataset.image_idx, self.train_dataset.num_images))
        p_bar = tqdm(total=len(val_data_loader))
        
        for iteration, data in enumerate(val_data_loader):
            # Map to device
            coords = data['coords'].to(self.device).squeeze(0)
            coords = val_dataset.transform_coords(coords, self.device)
            img_gts = data['rgb'].to(self.device).squeeze(0)
            pred = self.pipeline.nef(coords=coords, channels=["rgb"])[0]
            res = 1.0
            rgb_loss = ((pred - res * img_gts)**2).sum()
            
            val_dict['PSNR'] += [clamped_psnr(pred, img_gts)]
            val_dict['MSE'] += [rgb_loss.item()]
            val_dict['num_samples'] += [img_gts.size(0)]
            pred_image[val_dataset.shuffle_idx[coord_idx:coord_idx+coords.size(0)]] = pred.detach().cpu()
            coord_idx += coords.size(0)
            p_bar.update(1)
        p_bar.close()
        pred_image = torch.clamp(pred_image.reshape(H,W,3)*255,0.,255.).numpy().astype(np.uint8)
        gt_image = torch.clamp(val_dataset.data['orig_rgb'].reshape(H,W,3)*255,0.,255.).numpy().astype(np.uint8)
        val_dict['PSNR'] = sum(val_dict['PSNR'])/sum(val_dict['num_samples'])
        val_dict['MSE'] = sum(val_dict['MSE'])/sum(val_dict['num_samples'])
        if not self.metrics_only:
            val_dict['pred'] = pred_image
        
        if not self.train_dataset.static_coords and val_dict['MSE'] < self.best_state['rgb_loss']:
            self.best_state['rgb_loss'] = val_dict['MSE']
            self.best_state['PSNR'] = val_dict['PSNR']
            self.best_state['state_dict'] = copy.deepcopy(self.pipeline.state_dict())
            if not self.metrics_only:
                self.best_state['pred'] = val_dict['pred']
            if isinstance(self.pipeline.nef.grid, LatentGrid) or isinstance(self.pipeline.nef.grid, HashGrid) or isinstance(self.pipeline.nef.grid, CodebookOctreeGrid):
                self.best_state['total_size'] = self.log_dict['total_size']
                self.best_state['BPP'] = self.log_dict['total_size']*8000/math.prod(val_dataset.image_size)

        log_text = 'Validation Image {}/ {} EPOCH {}/{}'.format(self.train_dataset.image_idx, self.train_dataset.num_images, self.epoch, self.max_epochs)
        log_text += ' | PSNR: {:>.2E}'.format(val_dict['PSNR'])
        if isinstance(self.pipeline.nef.grid, LatentGrid) or isinstance(self.pipeline.nef.grid, HashGrid) or isinstance(self.pipeline.nef.grid, CodebookOctreeGrid):
            val_dict['BPP'] = self.log_dict['total_size']*8000/math.prod(val_dataset.image_size)
            log_text += ' | BPP: {:>.2E}'.format(self.log_dict['BPP'])
            log_text += ' | total size (kB): {:>.2E}'.format(self.log_dict['total_size'])
        log_text += ' | MSE: {:>.2E}'.format(val_dict['MSE'])
        if 'ent_loss' in self.log_dict:
            log_text += ' | ent loss: {:>.2E}'.format(self.log_dict['ent_loss'] / len(self.train_data_loader))
            # log_text += ' | codebook size (kB): {:>.3E}'.format(self.log_dict['net_kbytes_codebook'] / len(self.train_data_loader))
        
        log.info(log_text)

        if self.using_wandb:
            log_metric_to_wandb(f'image{image_idx}/validation/PSNR',val_dict['PSNR'], self.epoch)
            log_metric_to_wandb(f'image{image_idx}/validation/BPP',val_dict['BPP'], self.epoch)
            log_metric_to_wandb(f'image{image_idx}/validation/MSE',val_dict['MSE'], self.epoch)

        self.train_dataset.sample_mode = sample_mode_state
        # Resume state for SGA/Straight-through
        if self.extra_args["ldecode_enabled"]:
            latent_dec = self.pipeline.nef.grid.latent_dec
            if self.extra_args["use_sga"]:
                latent_dec.use_sga = use_sga_state
            if isinstance(latent_dec, MultiLatentDecoder):
                latent_dec.straight_through = straight_through_state
    
    def post_training(self):
        """
        Override this function to change the logic which runs after the last training iteration.
        This function runs once after training ends.
        """

        self.pipeline.load_state_dict(self.best_state['state_dict'])
        model_fname = os.path.join(self.log_dir_cur, f'model_best.pth')
        log.info(f'Saving best model checkpoint to: {model_fname}')
        if self.extra_args["model_format"] == "full":
            torch.save(self.pipeline, model_fname)
        else:
            torch.save(self.pipeline.state_dict(), model_fname)

        if not self.metrics_only:
            best_image = self.best_state['pred']
            H,W = self.train_dataset.image_size
            cv2.imwrite(os.path.join(self.log_dir_cur,'predicted.png'), cv2.cvtColor(best_image, cv2.COLOR_BGR2RGB))

        # Obtain model sizes and BPP for the best PSNR prediction 
        if isinstance(self.pipeline.nef.grid, LatentGrid) or \
            isinstance(self.pipeline.nef.grid, HashGrid) or \
            isinstance(self.pipeline.nef.grid, CodebookOctreeGrid):
            grid = self.pipeline.nef.grid
            size_ldec, size_latents = grid.size(use_torchac=True, use_prob_model=False)
            size_remainder = sum([p.numel()*torch.finfo(p.dtype).bits for n,p in self.pipeline.named_parameters() if 'grid' not in n])
            self.best_state['BPP'] = (size_latents+size_remainder+size_ldec)/math.prod(self.train_dataset.image_size)
            if size_ldec>0:
                self.best_state['ldec_size'] = size_ldec/8e3
            self.best_state['latent_size'] = size_latents/8e3
            self.best_state['remainder_size'] = size_remainder/8e3
            self.best_state['total_size'] = (size_latents+size_remainder+size_ldec)/8e3
            if isinstance(self.pipeline.nef.grid, LatentGrid):
                self.best_state['rounding_loss'] = torch.mean(torch.abs(grid.codebook - torch.round(grid.codebook))).item()

        state = self.scene_state.optimization
        torch.save(state, os.path.join(self.log_dir_cur, "scene_state.pth"))

        # Store only metrics
        if not self.metrics_only:
            del self.best_state['pred']
        del self.best_state['state_dict']
        with open(os.path.join(self.log_dir_cur,'metrics.json'), 'w') as f:
            json.dump(self.best_state, f)

    #######################
    # Logging
    #######################
        
    def log_cli(self):
        log_text = 'Image {}/ {} EPOCH {}/{}'.format(self.train_dataset.image_idx, self.train_dataset.num_images, self.epoch, self.max_epochs)
        log_text += ' | PSNR: {:>.2E}'.format(self.log_dict['PSNR'])
        if  isinstance(self.pipeline.nef.grid, LatentGrid) or \
            isinstance(self.pipeline.nef.grid, HashGrid) or \
            isinstance(self.pipeline.nef.grid, CodebookOctreeGrid):
            log_text += ' | BPP: {:>.2E}'.format(self.log_dict['BPP'])
            log_text += ' | total size (kB): {:>.2E}'.format(self.log_dict['total_size'])
        log_text += ' | total loss: {:>.2E}'.format(self.log_dict['total_loss'] / len(self.train_data_loader))
        log_text += ' | rgb loss: {:>.2E}'.format(self.log_dict['rgb_loss'] / len(self.train_data_loader))
        if 'ent_loss' in self.log_dict:
            log_text += ' | ent loss: {:>.2E}'.format(self.log_dict['ent_loss'] / len(self.train_data_loader))
            # log_text += ' | codebook size (kB): {:>.3E}'.format(self.log_dict['net_kbytes_codebook'] / len(self.train_data_loader))
        if isinstance(self.pipeline.nef.grid, LatentGrid) or isinstance(self.pipeline.nef.grid, HashGrid) or isinstance(self.pipeline.nef.grid, CodebookOctreeGrid):
            # if 'dec_size' in self.log_dict and isinstance(self.pipeline.nef.grid.latent_dec, MultiLatentDecoder):
            #     log_text += ' | dec size (kB): {:>.2E}'.format(self.log_dict['dec_size'])
            # log_text += ' | latent size (kB): {:>.2E}'.format(self.log_dict['latent_size'])
            log_text += ' | total size (kB): {:>.2E}'.format(self.log_dict['total_size'])
            if isinstance(self.pipeline.nef.grid, LatentGrid) and isinstance(self.pipeline.nef.grid.latent_dec, MultiLatentDecoder) or self.extra_args["use_sga"]:
                log_text += ' | temp: {:>.2E}'.format(self.pipeline.nef.grid.latent_dec.temperature)
            if self.extra_args["use_sga"]:
                log_text += ' | sga: {}'.format(self.pipeline.nef.grid.latent_dec.use_sga)
        
        log.info(log_text)

    def log_tb(self):
        """
        Override this function to change loss / other numeric logging to TensorBoard / Wandb.
        """
        image_idx = self.train_dataset.image_idx
        for key in self.log_dict:
            if 'loss' in key:
                self.writer.add_scalar(f'image{image_idx}/loss/{key}', self.log_dict[key] / len(self.train_data_loader), self.epoch)
                if self.using_wandb:
                    log_metric_to_wandb(f'image{image_idx}/loss/{key}', self.log_dict[key] / len(self.train_data_loader), self.epoch)
            if 'size' in key:
                self.writer.add_scalar(f'image{image_idx}/size/{key}', self.log_dict[key], self.epoch)
                if self.using_wandb:
                    log_metric_to_wandb(f'image{image_idx}/size/{key}', self.log_dict[key], self.epoch)

    def render_tb(self):
        image_idx = self.train_dataset.image_idx
        self.writer.add_image(f'image{image_idx}/pred', hwc_to_chw(self.best_state['pred']), self.epoch)
        if self.using_wandb:
            log_images_to_wandb(f'image{image_idx}/pred', hwc_to_chw(self.best_state['pred']), self.epoch)

    def save_state(self):
        """Save the current state of the training.
        """
        state = {
            "epoch": self.epoch,
            "image_idx": self.train_dataset.image_idx,
            "model": self.pipeline.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            'opt_state': self.scene_state.optimization
        }
        torch.save(state, os.path.join(self.log_dir, "resume_state.pth"))

    def resume_state(self):
        """Resume the training from the saved state.
        """
        if os.path.exists(os.path.join(self.log_dir, "resume_state.pth")):
            state = torch.load(os.path.join(self.log_dir, "resume_state.pth"))
            self.epoch = state["epoch"]+1
            self.pipeline.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.scene_state.optimization = state['opt_state']
            self.start_iteration = (self.epoch - 1) * self.iterations_per_epoch + 1
            log.info("Found saved state. Resuming training from epoch {}".format(self.epoch))
        else:
            log.info("No saved state found!")

    def is_first_iteration(self):
        return self.total_iterations == (self.start_iteration+1)