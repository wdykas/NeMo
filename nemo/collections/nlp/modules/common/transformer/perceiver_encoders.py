# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import copy

import torch

from nemo.collections.nlp.modules.common.transformer.transformer_decoders import TransformerDecoder
from nemo.collections.nlp.modules.common.transformer.transformer_encoders import TransformerEncoder
from nemo.collections.nlp.modules.common.transformer.transformer_modules import AttentionBridge
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.megatron_decoders import get_decoder_model
from nemo.collections.nlp.modules.common.megatron.megatron_encoders import get_encoder_model
from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults


try:
    from apex.transformer.enums import AttnMaskType

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False
    # fake missing classes with None attributes
    AttnMaskType = ApexGuardDefaults()

__all__ = ["PerceiverEncoder", "ParallelPerceiverEncoder"]


class PerceiverEncoder(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        inner_size: int,
        mask_future: bool = False,
        num_attention_heads: int = 1,
        attn_score_dropout: float = 0.0,
        attn_layer_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        hidden_act: str = "relu",
        pre_ln: bool = False,
        pre_ln_final_layer_norm: bool = True,
        hidden_steps: int = 32,
        hidden_init_method: str = "default",
        hidden_blocks: int = 2,
    ):
        super().__init__()

        self._hidden_steps = hidden_steps
        self._hidden_init_method = hidden_init_method
        self._hidden_blocks = hidden_blocks

        if self._hidden_init_method == "default":
            self._hidden_init_method = "params"

        if self.hidden_init_method not in self.supported_init_methods:
            raise ValueError(
                "Unknown hidden_init_method = {hidden_init_method}, supported methods are {supported_init_methods}".format(
                    hidden_init_method=self.hidden_init_method, supported_init_methods=self.supported_init_methods,
                )
            )

        diagonal = 0 if mask_future else None

        if self.hidden_init_method == "params":
            # learnable initial hidden values
            self.init_hidden = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(hidden_steps, hidden_size)))
            self.init_cross_att = TransformerDecoder(
                num_layers=1,
                hidden_size=hidden_size,
                inner_size=inner_size,
                num_attention_heads=num_attention_heads,
                attn_score_dropout=attn_score_dropout,
                attn_layer_dropout=attn_layer_dropout,
                ffn_dropout=ffn_dropout,
                hidden_act=hidden_act,
                pre_ln=pre_ln,
                pre_ln_final_layer_norm=pre_ln_final_layer_norm,
            )
            self.init_cross_att.diagonal = diagonal
        elif self.hidden_init_method == "bridge":
            # initialize latent with attention bridge
            self.att_bridge = AttentionBridge(hidden_size=hidden_size, k=hidden_steps, bridge_size=inner_size,)

        # cross-attention encoder
        layer = TransformerDecoder(
            num_layers=1,
            hidden_size=hidden_size,
            inner_size=inner_size,
            num_attention_heads=num_attention_heads,
            attn_score_dropout=attn_score_dropout,
            attn_layer_dropout=attn_layer_dropout,
            ffn_dropout=ffn_dropout,
            hidden_act=hidden_act,
            pre_ln=pre_ln,
            pre_ln_final_layer_norm=pre_ln_final_layer_norm,
        )
        layer.diagonal = diagonal
        self.cross_att_layers = torch.nn.ModuleList([copy.deepcopy(layer) for _ in range(hidden_blocks)])

        # self-attention encoder
        layer = TransformerEncoder(
            num_layers=num_layers,
            hidden_size=hidden_size,
            inner_size=inner_size,
            mask_future=mask_future,
            num_attention_heads=num_attention_heads,
            attn_score_dropout=attn_score_dropout,
            attn_layer_dropout=attn_layer_dropout,
            ffn_dropout=ffn_dropout,
            hidden_act=hidden_act,
            pre_ln=pre_ln,
            pre_ln_final_layer_norm=pre_ln_final_layer_norm,
        )
        self.self_att_layers = torch.nn.ModuleList([copy.deepcopy(layer) for _ in range(hidden_blocks)])

    @property
    def supported_init_methods(self):
        return ["params", "bridge"]

    @property
    def hidden_steps(self):
        return self._hidden_steps

    @property
    def hidden_blocks(self):
        return self._hidden_blocks

    @property
    def hidden_init_method(self):
        return self._hidden_init_method

    def forward(self, encoder_states, encoder_mask):
        """
        Args:
            encoder_states: output of the encoder (B x L_enc x H)
            encoder_mask: encoder inputs mask (B x L_enc)
        """
        # all hidden values are active
        hidden_mask = torch.ones(
            encoder_states.shape[0], self._hidden_steps, dtype=encoder_mask.dtype, device=encoder_mask.device
        )

        # initialize hidden state
        if self._hidden_init_method == "params":
            # initialize latent with learned parameters
            hidden_states = self.init_hidden.unsqueeze(0).expand(encoder_states.shape[0], -1, -1)
            hidden_states = self.init_cross_att(
                decoder_states=hidden_states,
                decoder_mask=hidden_mask,
                encoder_states=encoder_states,
                encoder_mask=encoder_mask,
            )
        elif self._hidden_init_method == "bridge":
            # initialize latent with attention bridge
            hidden_states = self.att_bridge(hidden=encoder_states, hidden_mask=encoder_mask,)

        # apply block (cross-attention, self-attention) multiple times
        # for block in range(self._hidden_blocks):
        for self_att, cross_att in zip(self.self_att_layers, self.cross_att_layers):
            residual = hidden_states

            # cross attention of hidden over encoder states
            hidden_states = cross_att(
                decoder_states=hidden_states,
                decoder_mask=hidden_mask,
                encoder_states=encoder_states,
                encoder_mask=encoder_mask,
            )

            # self-attention over hidden
            hidden_states = self_att(encoder_states=hidden_states, encoder_mask=hidden_mask,)

            # residual connection
            hidden_states += residual

        return hidden_states, hidden_mask



class ParallelPerceiverEncoder(MegatronModule):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        inner_size: int,
        num_attention_heads: int = 1,
        attn_score_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        hidden_act: str = "relu",
        hidden_steps: int = 32,
        hidden_init_method: str = "default",
        hidden_blocks: int = 2,
        precision: int = 16
    ):
        super().__init__()

        self._hidden_steps = hidden_steps
        self._hidden_init_method = hidden_init_method
        self._hidden_blocks = hidden_blocks

        if self._hidden_init_method == "default":
            self._hidden_init_method = "params"

        if self.hidden_init_method not in self.supported_init_methods:
            raise ValueError(
                "Unknown hidden_init_method = {hidden_init_method}, supported methods are {supported_init_methods}".format(
                    hidden_init_method=self.hidden_init_method, supported_init_methods=self.supported_init_methods,
                )
            )

        if self.hidden_init_method == "normal":
            self.init_hidden = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(hidden_steps, hidden_size)))
        elif self.hidden_init_method == "params":
            # learnable initial hidden values
            self.init_hidden = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(hidden_steps, hidden_size)))
            self.init_cross_att = get_decoder_model(
                arch="transformer",
                init_method=None,
                scaled_init_method=None,
                decoder_attn_mask_type=AttnMaskType.padding,
                hidden_size=hidden_size,
                num_layers=1,
                num_attention_heads=num_attention_heads,
                ffn_hidden_size=inner_size,
                hidden_dropout=ffn_dropout,
                attention_dropout=attn_score_dropout,
                precision=precision,
                activation=hidden_act,
                masked_softmax_fusion=False,
            )

        self.cross_att_layers = torch.nn.ModuleList()
        self.self_att_layers = torch.nn.ModuleList()

        for _ in range(hidden_blocks):
            # cross-attention encoder
            layer = get_decoder_model(
                arch="transformer",
                init_method=None,
                scaled_init_method=None,
                decoder_attn_mask_type=AttnMaskType.padding,
                hidden_size=hidden_size,
                num_layers=1,
                num_attention_heads=num_attention_heads,
                ffn_hidden_size=inner_size,
                hidden_dropout=ffn_dropout,
                attention_dropout=attn_score_dropout,
                precision=precision,
                activation=hidden_act,
                masked_softmax_fusion=False,
            )
            self.cross_att_layers.append(layer)

            # self-attention encoder
            layer = get_encoder_model(
                arch="transformer",
                init_method=None,
                scaled_init_method=None,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_attention_heads=num_attention_heads,
                ffn_hidden_size=inner_size,
                hidden_dropout=ffn_dropout,
                attention_dropout=attn_score_dropout,
                precision=precision,
                activation=hidden_act,
                masked_softmax_fusion=False,
            )
            self.self_att_layers.append(layer)

    @property
    def supported_init_methods(self):
        return ["normal", "params"]

    @property
    def hidden_steps(self):
        return self._hidden_steps

    @property
    def hidden_blocks(self):
        return self._hidden_blocks

    @property
    def hidden_init_method(self):
        return self._hidden_init_method

    def forward(self, encoder_states, encoder_mask):
        """
        Args:
            encoder_states: output of the encoder (B x L_enc x H)
            encoder_mask: encoder inputs mask (B x L_enc)
        """
        # all hidden values are active
        hidden_mask = torch.ones(
            encoder_states.shape[0], self._hidden_steps, dtype=encoder_mask.dtype, device=encoder_mask.device
        )

        # initialize hidden state
        if self._hidden_init_method == "normal":
            hidden_states = self.init_hidden.unsqueeze(0).expand(encoder_states.shape[0], -1, -1)
        elif self._hidden_init_method == "params":
            # initialize latent with learned parameters
            hidden_states = self.init_hidden.unsqueeze(0).expand(encoder_states.shape[0], -1, -1)
            hidden_states = self.init_cross_att(
                dec_input=hidden_states,
                dec_attn_mask=hidden_mask,
                enc_output=encoder_states,
                enc_output_mask=encoder_mask,
            )

        # apply block (cross-attention, self-attention) multiple times
        for self_att, cross_att in zip(self.self_att_layers, self.cross_att_layers):
            #residual = hidden_states

            # cross attention of hidden over encoder states
            hidden_states = cross_att(
                dec_input=hidden_states,
                dec_attn_mask=hidden_mask,
                enc_output=encoder_states,
                enc_output_mask=encoder_mask,
            )

            # self-attention over hidden
            hidden_states, hidden_mask = self_att(enc_input=hidden_states, enc_attn_mask=hidden_mask,)

            # residual connection
            #hidden_states += residual

        return hidden_states, hidden_mask