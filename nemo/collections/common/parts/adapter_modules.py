# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import torch
from dataclasses import dataclass, is_dataclass
from typing import Optional

from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn as nn

from nemo.collections.common.parts.utils import activation_registry
from nemo.core.classes.mixins import adapter_mixin_strategies


class AbstractAdapterModule(nn.Module):
    """
    Base class of Adapter Modules, providing common functionality to all Adapter Modules.
    """

    def setup_adapter_strategy(self, adapter_strategy: Optional[adapter_mixin_strategies.AbstractAdapterStrategy]):
        """
        Setup adapter strategy of this class, enabling dynamic change in the way the adapter output is
        merged with the input.

        When called successfully, will assign the variable `adapter_strategy` to the module.

        Args:
            adapter_strategy: Can be a None or an implementation of AbstractAdapterStrategy.
        """
        # set default adapter strategy
        if adapter_strategy is None:
            adapter_strategy = adapter_mixin_strategies.ResidualAddAdapterStrategyConfig()

        if is_dataclass(adapter_strategy):
            adapter_strategy = OmegaConf.structured(adapter_strategy)
            OmegaConf.set_struct(adapter_strategy, False)

        # The config must have the `_target_` field pointing to the actual adapter strategy class
        # which will load that strategy dynamically to this module.
        if isinstance(adapter_strategy, dict) or OmegaConf.is_config(adapter_strategy):
            self.adapter_strategy = instantiate(adapter_strategy)
        elif isinstance(adapter_strategy, adapter_mixin_strategies.AbstractAdapterStrategy):
            self.adapter_strategy = adapter_strategy
        else:
            raise AttributeError(f'`adapter_strategy` provided is invalid : {adapter_strategy}')


class LinearAdapter(AbstractAdapterModule):
    """
    Simple Linear Feedforward Adapter module with LayerNorm and singe hidden layer with activation function.
    Note: The adapter explicitly initializes its final layer with all zeros in order to avoid affecting the
    original model when all adapters are disabled.

    Args:
        in_features: Input dimension of the module. Note that for adapters, input_dim == output_dim.
        dim: Hidden dimension of the feed forward network.
        activation: Str name for an activation function.
        norm_position: Str, can be `pre` or `post`. Defaults to `post`. Determines whether the normalization
            will occur in the first layer or the last layer. Certain architectures may prefer one over the other.
        dropout: float value, whether to perform dropout on the output of the last layer of the adapter.
        adapter_strategy: By default, ResidualAddAdapterStrategyConfig. An adapter composition function object.
    """

    def __init__(
        self,
        in_features: int,
        dim: int,
        activation: str = 'swish',
        norm_position: str = "post",
        dropout: float = 0.0,
        adapter_strategy: adapter_mixin_strategies.ResidualAddAdapterStrategyConfig = None,
    ):
        super().__init__()

        activation = activation_registry[activation]()
        # If the activation can be executed in place, do so.
        if hasattr(activation, 'inplace'):
            activation.inplace = True

        assert norm_position in ['pre', 'post']
        self.norm_position = norm_position

        if norm_position == 'pre':
            self.module = nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Linear(in_features, dim, bias=False),
                activation,
                nn.Linear(dim, in_features, bias=False),
            )

        elif norm_position == 'post':
            self.module = nn.Sequential(
                nn.Linear(in_features, dim, bias=False),
                activation,
                nn.Linear(dim, in_features, bias=False),
                nn.LayerNorm(in_features),
            )

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        # Setup adapter strategy
        self.setup_adapter_strategy(adapter_strategy)

        # reset parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Final layer initializations must be 0
        if self.norm_position == 'pre':
            self.module[-1].weight.data *= 0

        elif self.norm_position == 'post':
            self.module[-1].weight.data *= 0
            self.module[-1].bias.data *= 0

    def forward(self, x):
        x = self.module(x)

        # Add dropout if available
        if self.dropout is not None:
            x = self.dropout(x)

        return x

    
    
class LoRA(AbstractAdapterModule):
    """
    Args:
        in_features: Input dimension of the module.
        out_features: Output dimension of the module.
        r (int, optional): The rank of the LoRA layer. Defaults to 8.
        alpha (int, optional): The hyperparameter used for scaling the LoRA reparametrization. Defaults to 8.
        dropout: float value, whether to perform dropout on the input.
        adapter_strategy: By default, AddAdapterStrategyConfig. An adapter composition function object.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: int = 8,
        dropout: float = 0.0,
        adapter_strategy: adapter_mixin_strategies.AddAdapterStrategyConfig = None,
    ):

        super().__init__()
        self.lora_A_q = nn.Parameter(torch.zeros(r,in_features))
        self.lora_B_q = nn.Parameter(torch.zeros(out_features,r))
        self.lora_A_k = nn.Parameter(torch.zeros(r,in_features))
        self.lora_B_k = nn.Parameter(torch.zeros(out_features,r))
        self.scaling = alpha / r
        
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        
        # Setup adapter strategy
        self.setup_adapter_strategy(adapter_strategy)
        
        # reset parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_q)
        nn.init.kaiming_uniform_(self.lora_A_k, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_k)
        
    def forward(self, x):
        
        # Add dropout if available
        if self.dropout is not None:
            x = self.dropout(x)
        
        x_q = x @ self.lora_A_q.T @ self.lora_B_q.T * self.scaling
        x_k = x @ self.lora_A_k.T @ self.lora_B_k.T * self.scaling
        return torch.cat((x_q,x_k), dim=2)
        
        
class Prefix(AbstractAdapterModule):
    """
    Args:
        in_features: Input dimension of the module.
        dim: Hidden dimension of the feed forward network.
        prefix_length: The length of the prefix tokens.
        dropout: float value, whether to perform dropout on the output of the last layer of the prefixes.
        adapter_strategy: By default, AddAdapterStrategyConfig. An adapter composition function object.
    """
    def __init__(
        self,
        in_features: int,
        dim: int,
        prefix_length: int,
        dropout: float = 0.0,
        adapter_strategy: adapter_mixin_strategies.AddAdapterStrategyConfig = None,
    ):

        super().__init__()
        
        self.in_features = in_features         
        self.prefix_length = prefix_length
        
        self.wte = nn.Embedding(self.prefix_length, in_features)
        self.control_trans = nn.Sequential(
            nn.Linear(in_features, dim),
            nn.Tanh(),
            nn.Linear(dim, 2 * in_features),
        )

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        
        # Setup adapter strategy
        self.setup_adapter_strategy(adapter_strategy)

    def forward(self, x):
        batch_size = x.shape[0]
        input_tokens = torch.arange(self.prefix_length).long()
        input_tokens = input_tokens.unsqueeze(0).expand(batch_size, -1).to(x.device)
        embs = self.wte(input_tokens)
        key_values = self.control_trans(embs)
        key_values = self.dropout(key_values)
        return key_values

@dataclass
class LinearAdapterConfig:
    in_features: int
    dim: int
    activation: str = 'swish'
    norm_position: str = 'post'
    dropout: float = 0.0
    adapter_strategy: Optional[dict] = adapter_mixin_strategies.ResidualAddAdapterStrategyConfig()
    _target_: str = "{0}.{1}".format(LinearAdapter.__module__, LinearAdapter.__name__)


@dataclass
class LoraConfig:
    in_features: int
    out_features: int
    r: int = 8
    alpha: int = 8
    dropout: float = 0.0
    adapter_strategy: Optional[dict] = adapter_mixin_strategies.AddAdapterStrategyConfig()
    _target_: str = "{0}.{1}".format(LoRA.__module__, LoRA.__name__)
    
    
@dataclass
class PrefixConfig:
    in_features: int
    dim: int
    prefix_length: int
    dropout: float = 0.0
    adapter_strategy: Optional[dict] = adapter_mixin_strategies.AddAdapterStrategyConfig()
    _target_: str = "{0}.{1}".format(Prefix.__module__, Prefix.__name__)