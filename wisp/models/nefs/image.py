# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
from typing import Dict, Any
from wisp.ops.geometric import sample_unif_sphere
from wisp.models.nefs import BaseNeuralField
from wisp.models.embedders import get_positional_embedder
from wisp.models.layers import get_layer_class
from wisp.models.activations import get_activation_class
from wisp.models.decoders import BasicDecoder
from wisp.models.grids import BLASGrid, HashGrid, LatentGrid

class NeuralImage(BaseNeuralField):
    """Model for encoding images as neural fields (RGB values).
    This implementation uses feature grids for a
    higher quality and more efficient implementation, following later trends in the literature,
    such as Neural Sparse Voxel Fields (Liu et al. 2020), Instant Neural Graphics Primitives (Muller et al. 2022)
    and Variable Bitrate Neural Fields (Takikawa et al. 2022).
    """

    def __init__(self,
                 grid: BLASGrid = None,
                 # embedder args
                 pos_embedder: str = 'none',
                 pos_multires: int = 10,
                 position_input: bool = False,
                 # decoder args
                 activation_type: str = 'relu',
                 final_activation: str = 'none',
                 layer_type: str = 'none',
                 hidden_dim: int = 128,
                 num_layers: int = 1,
                 ):
        """
        Creates a new NeuralImage instance, which maps 2D input coordinates to RGB

        This neural field consists of:
         * A feature grid (backed by an acceleration structure)
         * Color decoder
         * Optional: positional embedders for input position coords, concatenated to grid features.

         This neural field also supports:
          * Aggregation of multi-resolution features (more than one LOD) via summation or concatenation

        Args:
            grid: (BLASGrid): represents feature grids in Wisp. BLAS: "Bottom Level Acceleration Structure",
                to signify this structure is the backbone that captures
                a neural field's contents, in terms of both features and occupancy for speeding up queries.
                Notable examples: OctreeGrid, HashGrid, LatentGrid, TriplanarGrid, CodebookGrid.

            pos_embedder (str): Type of positional embedder to use for input coordinates.
                Options:
                 - 'none': No positional input is fed into the color decoder.
                 - 'identity': The sample coordinates are fed as is into the color decoder.
                 - 'positional': The sample coordinates are embedded with the Positional Encoding from
                    Mildenhall et al. 2020, before passing them into the color decoder.
            pos_multires (int): Number of frequencies used for 'positional' embedding of pos_embedder.
                 Used only if pos_embedder is 'positional'.
            position_input (bool): If True, the input coordinates will be passed into the decoder.
                 For 'positional': the input coordinates will be concatenated to the embedded coords.
                 For 'none' and 'identity': the embedder will behave like 'identity'.
            activation_type (str): Type of activation function to use in BasicDecoder:
                 'none', 'relu', 'sin', 'fullsort', 'minmax'.
            final_activation (str): Type of activation function to use after BasicDecoder:
                 'none', 'sigmoid', 'relu'.
            layer_type (str): Type of MLP layer to use in BasicDecoder:
                 'none' / 'linear', 'spectral_norm', 'frobenius_norm', 'l_1_norm', 'l_inf_norm'.
            hidden_dim (int): Number of neurons in hidden layers of both decoders.
            num_layers (int): Number of hidden layers in both decoders.
        """
        super().__init__()
        self.grid = grid

        # Init Embedders
        self.pos_embedder, self.pos_embed_dim = self.init_embedder(pos_embedder, pos_multires,
                                                                   include_input=position_input)

        # Init Decoder
        self.activation_type = activation_type
        self.final_activation = get_activation_class(final_activation)
        self.layer_type = layer_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.decoder_color = self.init_decoders(activation_type, layer_type, num_layers, hidden_dim)

        torch.cuda.empty_cache()

    def init_embedder(self, embedder_type, frequencies=None, include_input=False):
        """Creates positional embedding functions for the position.
        """
        if embedder_type == 'none' and not include_input:
            embedder, embed_dim = None, 0
        elif embedder_type == 'identity' or (embedder_type == 'none' and include_input):
            embedder, embed_dim = torch.nn.Identity(), 2    # Assumes pos input is always 2D
        elif embedder_type == 'positional':
            embedder, embed_dim = get_positional_embedder(frequencies=frequencies, 
                                                          input_dim=2,
                                                          include_input=include_input)
        else:
            raise NotImplementedError(f'Unsupported embedder type for NeuralImage: {embedder_type}')
        return embedder, embed_dim

    def init_decoders(self, activation_type, layer_type, num_layers, hidden_dim):
        """Initializes the decoder object.
        """
        decoder_color = BasicDecoder(input_dim=self.color_net_input_dim(),
                                     output_dim=3,
                                     activation=get_activation_class(activation_type),
                                     bias=True,
                                     layer=get_layer_class(layer_type),
                                     num_layers=num_layers + 1,
                                     hidden_dim=hidden_dim,
                                     skip=[])
        return decoder_color

    def register_forward_functions(self):
        """Register the forward functions.
        """
        self._register_forward_function(self.rgb, ["rgb"])

    def rgb(self, coords, lod_idx=None):
        """Compute color for the provided coordinates.

        Args:
            coords (torch.FloatTensor): tensor of shape [batch, 2]
            lod_idx (int): index into active_lods. If None, will use the maximum LOD.
        
        Returns:
            {"rgb": torch.FloatTensor}:
                - RGB tensor of shape [batch, 3]
        """
        if lod_idx is None:
            lod_idx = len(self.grid.active_lods) - 1
        batch, _ = coords.shape

        # Embed coordinates into high-dimensional vectors with the grid.
        feats = self.grid.interpolate(coords, lod_idx).reshape(batch, self.effective_feature_dim())

        # Optionally concat the positions to the embedding
        if self.pos_embedder is not None:
            embedded_pos = self.pos_embedder(coords).view(batch, self.pos_embed_dim)
            feats = torch.cat([feats, embedded_pos], dim=-1)

        # Colors are values [0, 1] floats (if final activation clips)
        # colors ~ (batch, 3)
        colors = self.final_activation(self.decoder_color(feats))

        return dict(rgb=colors)

    def effective_feature_dim(self):
        if self.grid.multiscale_type == 'cat':
            effective_feature_dim = self.grid.feature_dim * self.grid.num_lods
        else:
            effective_feature_dim = self.grid.feature_dim
        return effective_feature_dim

    def color_net_input_dim(self):
        return self.effective_feature_dim() + self.pos_embed_dim

    def public_properties(self) -> Dict[str, Any]:
        """ Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        properties = {
            "Grid": self.grid,
            "Pos. Embedding": self.pos_embedder,
            "Decoder (color)": self.decoder_color,
            "Final Activation": self.final_activation
        }
        return properties
