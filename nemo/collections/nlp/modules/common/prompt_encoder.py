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

from typing import Dict, List, Optional

import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import Trainer

from omegaconf.dictconfig import DictConfig
from nemo.core.classes import Exportable, NeuralModule
from nemo.core.classes.common import typecheck
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.models.language_modeling.megatron_bert_model import MegatronBertModel
from nemo.collections.nlp.modules.common.transformer.perceiver_encoders import ParallelPerceiverEncoder
from nemo.collections.nlp.modules.common.megatron.transformer import ParallelMLP
from nemo.collections.nlp.modules.common.megatron.utils import init_method_normal, scaled_init_method_normal

__all__ = ['PromptEncoder']


class PromptEncoder(MegatronModule):
    """
    The Prompt Encoder network that is used to generate the virtual token embeddings
    """

    def __init__(
            self, prompt_seq_len: int, hidden_size: int, prompt_dropout: float, num_layers: int,
            reparametrize: bool, prompt_gen_type: str, precision: int,
            trainer: Optional[Trainer] = None, restore_from_path: str = None, prompt_enc_cfg: DictConfig = None
    ):
        """
        Initializes the PromptEncoder module.
        Args:
            prompt_seq_len: number of prompt embeddings to produce
            hidden_size: hidden dimension
            prompt_dropout: the dropout used for the LSTM
            num_layers: number of layers used in the LSTM
            reparametrize: whether to parametrize embeddings
            prompt_gen_type: type of parametrization model
        """
        super().__init__()
        self.prompt_seq_len = prompt_seq_len
        self.hidden_size = hidden_size
        self.reparametrize = reparametrize
        PromptGenModels = {
            "sequential": SeqPromptGenerator,
            "linear": LinearPromptGenerator,
            "bottleneck": BottleneckPromptGenerator,
            "bert": BertPromptGenerator,
            "adaptivepooling": AdaptivePoolingPromptGenerator
        }
        self.prompt_gen_type = prompt_gen_type
        if prompt_gen_type in ["bottleneck", "bert", "adaptivepooling"]:
            assert self.reparametrize
        if prompt_gen_type in ["sequential", "linear"]:
            # ent embedding
            self.cloze_mask = torch.LongTensor([1] * prompt_seq_len).bool()
            self.register_buffer('seq_indices', torch.LongTensor(list(range(self.prompt_seq_len))))
            self.embedding = torch.nn.Embedding(self.prompt_seq_len, self.hidden_size)
        if self.reparametrize:
            PromptGen = PromptGenModels[prompt_gen_type]
            self.prompt_generator = PromptGen(
                hidden_size, prompt_dropout, prompt_seq_len, num_layers, precision,
                trainer, restore_from_path, prompt_enc_cfg
            )

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, prompt_condition=None, prompt_condition_mask=None) -> torch.Tensor:
        if self.prompt_gen_type in ["sequential", "linear"]:
            prompt_embeds = self.embedding(self.seq_indices).unsqueeze(0)
            if prompt_condition is not None:
                bz, task_seq, _ = prompt_condition.shape
                _, seq, emb = prompt_embeds.shape
                prompt_embeds = prompt_embeds.expand(bz, seq, emb).clone()
                length = min(task_seq, seq)
                prompt_embeds[:, 0:length, :] = prompt_condition[:, 0:length, :]
        else:
            prompt_embeds = prompt_condition
        if self.reparametrize:
            prompt_embeds = self.prompt_generator(prompt_embeds, prompt_condition_mask)
        return prompt_embeds


class LinearPromptGenerator(MegatronModule):
    def __init__(
            self,
            hidden_size: int,
            prompt_dropout: float,
            prompt_seq_len: int,
            num_layers: int,
            precision: int,
            trainer: Trainer,
            restore_from_path: str,
            prompt_enc_cfg: DictConfig

    ):
        """
        Initializes the PromptEncoder module.
        Args:
            hidden_size: hidden dimension
            lstm_dropout: the dropout used for the LSTM
            num_layers: number of layers used in the LSTM
        """
        super().__init__()
        self.prompt_seq_len = prompt_seq_len
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, prompt_embeds, prompt_embeds_mask) -> torch.Tensor:
        prompt_embeds = self.mlp_head(prompt_embeds[:, 0:self.prompt_seq_len, :])
        return prompt_embeds


class SeqPromptGenerator(MegatronModule):
    def __init__(
            self,
            hidden_size: int,
            prompt_dropout: float,
            prompt_seq_len: int,
            num_layers: int,
            precision: int,
            trainer: Trainer,
            restore_from_path: str,
            prompt_enc_cfg: DictConfig
    ):
        """
        Initializes the PromptEncoder module.
        Args:
            prompt_seq_len: number of prompt embeddings to produce
            hidden_size: hidden dimension
            prompt_dropout: hidden dropout
            num_layers: number of layers
        """
        super().__init__()
        self.prompt_seq_len = prompt_seq_len
        self.lstm_head = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=num_layers,
            dropout=prompt_dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), nn.Linear(hidden_size, hidden_size)
        )

    def forward(self, prompt_embeds, prompt_embeds_mask) -> torch.Tensor:
        prompt_embeds = self.mlp_head(self.lstm_head(prompt_embeds)[0][:, -self.prompt_seq_len:, :])
        return prompt_embeds


class BertPromptGenerator(MegatronModule):
    def __init__(
            self,
            hidden_size: int,
            prompt_dropout: float,
            prompt_seq_len: int,
            num_layers: int,
            precision: int,
            trainer: Trainer,
            restore_from_path: str,
            prompt_enc_cfg: DictConfig,
    ):
        """
        Initializes the PromptEncoder module.
        Args:
            prompt_seq_len: number of prompt embeddings to produce
            hidden_size: hidden dimension
            prompt_dropout: hidden dropout
            num_layers: number of layers
        """
        super().__init__()
        self.prompt_seq_len = prompt_seq_len
        model = MegatronBertModel.restore_from(
            restore_from_path,
            trainer=trainer,
            override_config_path=prompt_enc_cfg
        )
        bert_model = model.bert_model
        #TODO: I can't seem to load a nemo model. When this is fixed, below needs to be updated
        if hasattr(prompt_enc_cfg, 'new_pos_emb_size'):
            new_num_tokens = prompt_enc_cfg.new_pos_emb_size
            old_embeddings = bert_model.embeddings.position_embeddings
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
            old_pos_ids = bert_model.embeddings.position_ids

            bert_model.position_ids = torch.arange(
                new_num_tokens, dtype=torch.long, device=old_pos_ids.device
            ).unsqueeze(0)
            new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
            new_embeddings.to(old_embeddings.weight.device, dtype=new_embeddings.weight.dtype)
            n = min(old_num_tokens, new_num_tokens)
            new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]
            bert_model.embeddings.position_embeddings = new_embeddings

        self.word_embeddings = bert_model.embeddings.word_embeddings
        self.position_embeddings = bert_model.embeddings.position_embeddings
        self.tokentype_embeddings = bert_model.embeddings.token_type_embeddings
        self.embedding_dropout = bert_model.embeddings.dropout
        self.position_ids = bert_model.position_ids
        self.tokentype_ids = torch.zeros_like(self.position_ids)
        self.encoder = bert_model.encoder

    def get_extended_attention_mask(self, attention_mask, input_shape) -> torch.Tensor:
        #TODO: remove when we are able to load nemo models
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, prompt_embeds, prompt_embeds_mask) -> torch.Tensor:
        position_ids = self.position_ids[:, :prompt_embeds.size(1)].repeat(prompt_embeds.size(0), 1)
        tokentype_ids = self.tokentype_ids[:, :prompt_embeds.size(1)].repeat(prompt_embeds.size(0), 1)
        prompt_embeds = prompt_embeds + self.position_embeddings(position_ids) + self.tokentype_embeddings(tokentype_ids)
        prompt_embeds = self.embedding_dropout(prompt_embeds)
        prompt_embeds_mask = self.get_extended_attention_mask(
            prompt_embeds_mask, prompt_embeds.size()[:-1]
        )
        outputs = self.encoder(
            hidden_states=prompt_embeds, attention_mask=prompt_embeds_mask
        )
        prompt_embeds = outputs.last_hidden_state
        return prompt_embeds[:, :self.prompt_seq_len]


class BottleneckPromptGenerator(MegatronModule):
    def __init__(
            self,
            hidden_size: int,
            prompt_dropout: float,
            prompt_seq_len: int,
            num_layers: int,
            precision: int,
            trainer: Trainer,
            restore_from_path: str,
            prompt_enc_cfg: DictConfig,
    ):
        """
        Initializes the PromptEncoder module.
        Args:
            prompt_seq_len: number of prompt embeddings to produce
            hidden_size: hidden dimension
            prompt_dropout: hidden dropout
            num_layers: number of layers
        """
        super().__init__()
        self.prompt_seq_len = prompt_seq_len
        self.encoder = ParallelPerceiverEncoder(
                num_layers=num_layers,
                hidden_size=hidden_size,
                inner_size=hidden_size * 4,
                num_attention_heads=16,
                attn_score_dropout=prompt_dropout,
                ffn_dropout=prompt_dropout,
                hidden_act='gelu',
                hidden_steps=prompt_seq_len,
                hidden_blocks=1,
                hidden_init_method="default",
                precision=precision
            )

    def forward(self, prompt_embeds, prompt_embeds_mask) -> torch.Tensor:
        prompt_embeds, mask = self.encoder(prompt_embeds, prompt_embeds_mask)
        return prompt_embeds


class AdaptivePoolingPromptGenerator(MegatronModule):
    def __init__(
            self,
            hidden_size: int,
            prompt_dropout: float,
            prompt_seq_len: int,
            num_layers: int,
            precision: int,
            trainer: Trainer,
            restore_from_path: str,
            prompt_enc_cfg: DictConfig,
            init_method_std: int = 0.02
    ):
        """
        Initializes the PromptEncoder module.
        Args:
            prompt_seq_len: number of prompt embeddings to produce
            hidden_size: hidden dimension
            prompt_dropout: hidden dropout
            num_layers: number of layers
        """
        super().__init__()
        self.prompt_seq_len = prompt_seq_len
        self.max_seq_len = prompt_seq_len * num_layers
        self.prompt_dropout = prompt_dropout

        self.encoder = nn.ModuleList()
        for i in range(num_layers):
            self.encoder.append(ParallelMLP(
                init_method=init_method_normal(init_method_std),
                output_layer_init_method=scaled_init_method_normal(init_method_std, num_layers),
                hidden_size=hidden_size,
                ffn_hidden_size=hidden_size,
                bias_gelu_fusion=False
            ))
            self.encoder.append(nn.AdaptiveAvgPool1d(self.max_seq_len // (i+1)))
        self.final_layer = ParallelMLP(
                init_method=init_method_normal(init_method_std),
                output_layer_init_method=scaled_init_method_normal(init_method_std, num_layers),
                hidden_size=hidden_size,
                ffn_hidden_size=hidden_size,
                bias_gelu_fusion=False
        )

    def set_prompt_dropout(self, p):
        self.prompt_dropout = p

    def forward(self, prompt_embeds, prompt_embeds_mask) -> torch.Tensor:
        for encoder_layer in self.encoder:
            prompt_embeds = encoder_layer(prompt_embeds)
            if isinstance(prompt_embeds, tuple):
                prompt_embeds = prompt_embeds[0] + prompt_embeds[1]
                prompt_embeds = F.dropout(prompt_embeds, p=self.prompt_dropout, training=self.training)
            prompt_embeds = prompt_embeds.transpose(1, 2)
        prompt_embeds = self.final_layer(prompt_embeds)
        prompt_embeds = prompt_embeds[0] + prompt_embeds[1]
        return prompt_embeds