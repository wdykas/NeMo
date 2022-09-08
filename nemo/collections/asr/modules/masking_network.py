# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections import OrderedDict

import torch
import torch.distributed
import torch.nn as nn

from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.collections.asr.modules.transformer_encoder import TransformerEncoder
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder, FusionLayer
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import EncodedRepresentation, NeuralType, SpectrogramType
from nemo.core.neural_types.elements import MaskType
from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling

__all__ = ['MaskingNetwork']


speakerbeam_activations = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "selu": nn.SELU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
}

EPS = 1e-8
class MaskingNetwork(NeuralModule, Exportable):
    """
    SpeakerBeam
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8736286

    Args:
        feat_in (int): the size of feature channels
        n_layers (int): number of layers of ConformerBlock
        d_model (int): the hidden size of the model
        feat_out (int): the size of the output features
            Defaults to -1 (means feat_out is d_model)
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "features": NeuralType(('B', 'D'), EncodedRepresentation()),
            }
        )

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return OrderedDict({"audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),})

    def __init__(
        
        
        self,
        feat_in,
        d_model,
        feat_out,
        intra_model,
        subsampling_factor=4,
        subsampling="striding",
        use_intra_target_embed=False, 
        emb_dim=256,
        fusion_type='add',
    ):
        super().__init__()


        self.norm = nn.GroupNorm(1, feat_in, eps=EPS)
        
        if subsampling_conv_channels == -1:
            subsampling_conv_channels = d_model
        self.pre_encode = ConvSubsampling(
            subsampling=subsampling,
            subsampling_factor=subsampling_factor,
            feat_in=feat_in,
            feat_out=d_model,
            conv_channels=subsampling_conv_channels,
            activation=nn.ReLU(),
            is_causal=False,
                )
        self.use_intra_target_embed = use_intra_target_embed

        if intra_model.get('model_type', None) == 'transformer':
            self.encoder = TransformerEncoder(
                n_layers=intra_model['num_layers'],
                d_model=intra_model['d_model'],
                ff_expansion_factor=intra_model['ff_expansion_factor'],
                self_attention_model=intra_model['pos_encoding'],
                xscaling=intra_model['x_scaling'],
                n_heads=intra_model['n_heads'],
                pre_norm=intra_model['pre_norm'],
                dropout=intra_model['dropout'],
                dropout_emb=intra_model['dropout_emb'],
                dropout_att=intra_model['dropout_att'],
            )
        elif intra_model.get('model_type', None) == 'conformer':
            self.encoder = ConformerEncoder(
                n_layers=intra_model['num_layers'],
                d_model=intra_model['d_model'],
                ff_expansion_factor=intra_model['ff_expansion_factor'],
                self_attention_model=intra_model['pos_encoding'],
                xscaling=intra_model['x_scaling'],
                n_heads=intra_model['n_heads'],
                pre_norm=intra_model['pre_norm'],
                dropout=intra_model['dropout'],
                dropout_emb=intra_model['dropout_emb'],
                dropout_att=intra_model['dropout_att'],
            )
        self._feat_in = feat_in
        self._feat_out = feat_out

       
        self.fusion_mdl.append(FusionLayer(feat_out, emb_dim, fusion_type))


    @typecheck()
    def forward(self, x, x_len, emb):
      
        x = self.norm(x)
        x, x_len = self.pre_encode(x=x, lengths=x_len)
        x_used_with_emb = self.fusion_mdl(x, emb)



