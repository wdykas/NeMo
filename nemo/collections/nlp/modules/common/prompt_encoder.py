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
from pytorch_lightning import Trainer

from nemo.core.classes import Exportable, NeuralModule
from nemo.core.classes.common import typecheck
from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.models.language_modeling.bert_lm_model import BERTLMModel
from nemo.collections.nlp.modules.common.transformer.perceiver_encoders import ParallelPerceiverEncoder

__all__ = ['PromptEncoder']


class PromptEncoder(MegatronModule):
    """
    The Prompt Encoder network that is used to generate the virtual token embeddings
    """

    def __init__(
            self, prompt_seq_len: int, hidden_size: int, prompt_dropout: float, num_layers: int,
            reparametrize: bool, prompt_gen_type: str, precision: int, trainer: Optional[Trainer] = None
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
            "bert": BertPromptGenerator
        }
        self.prompt_gen_type = prompt_gen_type
        if prompt_gen_type in ["bottleneck", "bert"]:
            assert self.reparametrize
        if prompt_gen_type in ["sequential", "linear"]:
            # ent embedding
            self.cloze_mask = torch.LongTensor([1] * prompt_seq_len).bool()
            self.register_buffer('seq_indices', torch.LongTensor(list(range(self.prompt_seq_len))))
            self.embedding = torch.nn.Embedding(self.prompt_seq_len, self.hidden_size)
        if self.reparametrize:
            PromptGen = PromptGenModels[prompt_gen_type]
            self.prompt_generator = PromptGen(
                hidden_size, prompt_dropout, prompt_seq_len, num_layers, precision, trainer
            )

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
            trainer: Trainer
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
            trainer: Trainer
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
            trainer: Trainer
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
        self.encoder = BERTLMModel.restore_from(
            "bertbaseuncased",
            trainer=trainer
        )

    def forward(self, prompt_embeds, prompt_embeds_mask) -> torch.Tensor:
        prompt_embeds, mask = self.encoder(prompt_embeds, prompt_embeds_mask)
        return prompt_embeds[:, :self.prompt_seq_len]


class BottleneckPromptGenerator(MegatronModule):
    def __init__(
            self,
            hidden_size: int,
            prompt_dropout: float,
            prompt_seq_len: int,
            num_layers: int,
            precision: int,
            trainer: Trainer
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
                num_attention_heads=12,
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