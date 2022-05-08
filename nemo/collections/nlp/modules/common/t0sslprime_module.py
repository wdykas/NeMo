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

import numpy as np

import torch
from torch import nn
import torch.distributed as dist
from nemo.core.classes import Exportable, NeuralModule

from nemo.utils.app_state import AppState

__all__ = ['AttentiveMaxPooling', 'SeqProjectionNetwork']


class AttentiveMaxPooling(NeuralModule, Exportable):

    def __init__(self, hidden_size):
        super().__init__()
        self.linear_trf = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden, attn_mask):
        hidden = self.linear_trf(hidden)
        k = hidden.permute(0, 2, 1)
        k = k/k.norm(dim=1).unsqueeze(1)  # normalized key
        q = hidden.max(dim=1)[0].unsqueeze(1)  # max pooled query
        attn = torch.bmm(q, k)
        if attn_mask is not None:
            #TODO: need inf mask
            attn += attn_mask[:, :1, :]
        attn = torch.softmax(attn, dim=2)
        scores = torch.matmul(attn, hidden).squeeze(1)
        return scores


class SeqProjectionNetwork(NeuralModule, Exportable):
    def __init__(self, hidden_size, num_virtual_classes, nlayers=3, use_ln=False):
        super().__init__()

        self.attn_max_pool = AttentiveMaxPooling(hidden_size)
        layers = []
        for _ in range(nlayers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            if use_ln:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_size, num_virtual_classes, bias=False)

    def forward(self, hidden_seq, attn_mask=None):
        x = self.attn_max_pool(hidden_seq, attn_mask)
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        x = self.output_layer(x)
        return x

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False


class SSLPromptLoss(NeuralModule, Exportable):
    def __init__(
            self,
            num_virtual_classes,
            teacher_temp_init,
            teacher_temp_final,
            epochs,
            student_temp=0.1,
            center_momentum=0.9
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, num_virtual_classes))
        self.teacher_temp_schedule = np.linspace(teacher_temp_init, teacher_temp_final, epochs)

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = nn.functional.softmax((teacher_output.detach() - self.center.detach()) / temp, dim=-1)
        loss = torch.sum(-teacher_out * nn.functional.log_softmax(student_out, dim=-1), dim=-1).mean()
        return loss

    @torch.no_grad()
    def update_center(self, teacher_out1, teacher_out2):
        """
        Update center used for teacher output.
        """
        app_state = AppState()
        teacher_output = torch.concat((teacher_out1, teacher_out2), dim=0)
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)  # TODO: how does this behave for partitioned models + multi-node
        world_size = app_state.world_size
        batch_center = batch_center / (len(teacher_output) * world_size)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)