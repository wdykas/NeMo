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

from abc import ABC
from dataclasses import dataclass, is_dataclass
from typing import List, Optional, Union

import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.utils import logging, model_utils

# Global registry of all loras
LORA_REGISTRY = {}


@dataclass
class LoraRegistryInfo:
    base_class: type
    lora_class: type

    # generated automatically
    base_class_path: str = ""
    lora_class_path: str = ""

    def __post_init__(self):
        self.base_class_path = f'{self.base_class.__module__}.{self.base_class.__name__}'
        self.lora_class_path = f'{self.lora_class.__module__}.{self.lora_class.__name__}'


def register_lora(base_class: type, lora_class: type):
    """
    Registers a pair (Base class, Lora class) into the lora registry, used for de-referencing.

    Args:
        base_class: A Class, which is the base class of the object.
        lora_class: A Class, which is the subclass of the base class, and implements the Lora mixin methods.
    """
    global LORA_REGISTRY
    base_class_path = f'{base_class.__module__}.{base_class.__name__}'
    lora_class_path = f'{lora_class.__module__}.{lora_class.__name__}'

    # test if base class already in registry
    if base_class_path in LORA_REGISTRY:
        raise ValueError(f"`{base_class_path}` has already been added to the lora registry !")

    # test if lora is a subclass of the base class
    if not issubclass(lora_class, base_class):
        raise ValueError(f"`{lora_class_path}` is not a sub-class of {base_class_path} !")

    # register the base class : lora class pair
    LORA_REGISTRY[base_class_path] = LoraRegistryInfo(base_class=base_class, lora_class=lora_class)

    # attach lora class to base class
    base_class._meta_lora_class = lora_class

    # attach base class to lora class
    lora_class._meta_base_class = base_class


def get_registered_lora(cls: Union[str, type]) -> Optional[LoraRegistryInfo]:
    """
    Resolves a provided `cls` (whether str path to class, a registered base or an lora class)
    to obtain the metadata for the lora.

    Args:
        cls: Can be a str (absolute path to a class), a base class or an lora class (which have already
            been registered).

    Returns:
        A LoraRegistryInfo object if it could resolve successfully, otherwise None.
    """
    global LORA_REGISTRY
    if isinstance(cls, str):
        cls = model_utils.import_class_by_path(cls)

    # If an lora class was provided, de-reference its base class
    if hasattr(cls, '_meta_base_class'):
        cls = cls._meta_base_class

    class_path = f'{cls.__module__}.{cls.__name__}'

    # If base class, check registry
    if class_path in LORA_REGISTRY:
        return LORA_REGISTRY[class_path]

    return None


def _prepare_default_lora_config(*, global_key: str, meta_key: str, cfg: DictConfig = None) -> DictConfig:
    if cfg is None:
        cfg = OmegaConf.create({})

    with open_dict(cfg):
        if global_key not in cfg:
            cfg[global_key] = OmegaConf.create({})

        if meta_key not in cfg[global_key]:
            cfg[global_key][meta_key] = OmegaConf.create({})

        if 'modules' not in cfg[global_key][meta_key]:
            cfg[global_key][meta_key]['modules'] = OmegaConf.create({})

    return cfg


class LoraModuleMixin(ABC):
    """ Generic Lora Mixin that can augment any torch.nn.Module with Lora module support.

    This mixin class adds a hierarchical way to add any type of Lora modules to a pre-existing module.
    Since Models are inherently also nn.Module, this mixin can be attached to any Model or Module.
    This mixin class adds several utility methods which are utilized or overridden as necessary.

    An Lora module is any Pytorch nn.Module that possess a few properties :

        -   It's input and output dimension are the same, while the hidden dimension need not be the same.
        -   The final layer of the Lora module is zero-initialized, so that the residual connection to the lora
                yields the original output.

    This mixin adds the following instance variables to the class this inherits it:

        -   `lora_layer`: A torch.nn.ModuleDict(), whose keys are the names of the lora (globally unique),
                and values are the Lora nn.Module().
        -   `lora_cfg`: A OmegaConf DictConfig object that holds the config of the loras that are initialized.
        -   `lora_name`: A str resolved name which is unique key globally, but more than one modules may share
                this name.
        -   `lora_global_cfg_key`: A str representing a key in the model config that can be provided by the user.
                The value resolves to `global_cfg`, and can be overridden via `model.cfg.loras.global_cfg.*`.
        -   `lora_metadata_cfg_key`: A str representing a key in the model config that is used to preserve the
                metadata of the lora config.

    **Note**: This module is **not** responsible for maintaining its config. Subclasses must ensure config is updated
        or preserved as needed. It is the responsibility of the subclasses to propagate the most up to date config to
        lower layers.
    """

    lora_global_cfg_key = "global_cfg"
    lora_metadata_cfg_key = "lora_meta_cfg"

    def add_lora(self, name: str, cfg: DictConfig):
        """
        Add an Lora module to this module.

        Args:
            name: A globally unique name for the lora. Will be used to access, enable and disable loras.
            cfg: A DictConfig or Dataclass that contains at the bare minimum `__target__` to instantiate a
                new Lora module.
        """
        # Convert to DictConfig from dict or Dataclass
        if is_dataclass(cfg):
            cfg = OmegaConf.structured(cfg)

        if not isinstance(cfg, DictConfig):
            cfg = DictConfig(cfg)

        # Add lora_layer ModuleDict() if not present.
        if not hasattr(self, 'lora_layer'):
            self.lora_layer = nn.ModuleDict()

        # Add lora_cfg if it doesnt exist or hasnt been assigned yet.
        if not hasattr(self, 'lora_cfg'):
            self.lora_cfg = OmegaConf.create({})

        # Resolve the module name and lora name (if module name is provided)
        _, lora_name = self.resolve_lora_module_name_(name)

        # Add lora_name to this module for later identification
        self.lora_name = lora_name

        # Assert that name is globally unique to all loras.
        if lora_name in self.lora_layer:
            raise ValueError(
                f"Lora with name `{name}` already exists ! Lora names = {list(self.lora_layer.keys())}"
            )

        # Assert that name is not `lora_global_cfg_key`
        if lora_name == self.lora_global_cfg_key:
            raise ValueError(f"Loras cannot have the reserved name : `{self.lora_global_cfg_key}`")

        # Update internal config and instantiate the Lora module
        with open_dict(cfg), open_dict(self.lora_cfg):
            lora_enabled = cfg.pop('enabled', True)
            self.lora_layer[lora_name] = instantiate(cfg)

            cfg['enabled'] = lora_enabled
            self.lora_cfg[lora_name] = cfg

    def is_lora_available(self) -> bool:
        """
        Checks if any Lora module has been instantiated.

        Returns:
            bool, determining if any Lora module has been instantiated. Returns true even if the loras are
            enabled or disabled, false only if no loras exist.
        """
        if hasattr(self, 'lora_layer'):
            return self.lora_layer is not None and len(self.lora_layer) > 0
        return False

    def set_enabled_loras(self, name: Optional[str] = None, enabled: bool = True):
        """
        Updated the internal lora config, determining if an lora (or all loras) are either
        enabled or disabled.

        A common user pattern would be to disable all loras (either after adding them, or restoring a model
        with pre-existing loras) and then simply enable one of the loras.

        .. code::

            module.set_enabled_loras(enabled=False)
            module.set_enabled_loras(name=<some lora name>, enabled=True)

        Args:
            name: Optional str. If a str name is given, the config will be updated to the value of `enabled`.
                If no name is given, then all loras will be enabled/disabled.
            enabled: Bool, determines if the lora(s) will be enabled/disabled.
        """
        if not self.is_lora_available():
            raise ValueError("No lora is available to enable/disable")

        # If name is None, enable/disable all loras.
        if name is None:
            for key, config in self.lora_cfg.items():
                # Skip the global lora config
                if key == self.lora_global_cfg_key:
                    continue

                # Enable/Disable the current lora
                self.lora_cfg[key]['enabled'] = enabled
        else:
            _, lora_name = self.resolve_lora_module_name_(name)

            # Cannot set the state of the global config for loras
            if lora_name == self.lora_global_cfg_key:
                raise ValueError(
                    f'Cannot set the state of the global config of loras, '
                    f'given name = `{self.lora_global_cfg_key}`'
                )

            # Enable/Disable just named lora
            self.lora_cfg[lora_name]['enabled'] = enabled

    def get_enabled_loras(self) -> List[str]:
        """
        Returns a list of all enabled loras names. The names will always be the resolved names, without
        module info.

        Returns:
            A list of str names of each enabled lora names(s).
        """
        if not self.is_lora_available():
            return []

        # populate set of available modules (by name)
        available_module_names = set([])
        if hasattr(self, 'lora_layer'):
            available_module_names.update(list(self.lora_layer.keys()))

        enabled_loras = []
        for name, config in self.lora_cfg.items():
            # Skip the global lora config
            if name == self.lora_global_cfg_key:
                continue

            # If name is in the current available modules, and it is enabled in the config
            if name in available_module_names and self.lora_cfg[name]['enabled']:
                enabled_loras.append(name)

        return enabled_loras

    # Inherited methods that dont need to be overridden

    def unfreeze_enabled_loras(self, freeze_batchnorm: bool = True) -> None:
        """
        Utility method to unfreeze only the enabled Lora module(s).

        A common user pattern is to freeze all the modules (including all the loras), and then
        unfreeze just the required loras.

        .. code::

            module.freeze()  # only available to nemo.core.NeuralModule !
            module.unfreeze_enabled_loras()

        Args:
            freeze_batchnorm: An optional (and recommended) practice of freezing the updates to the moving average
                buffers of any and all BatchNorm*D layers. This is necessary to ensure that disabling all loras
                will precisely yield the original (base) model's outputs.
        """
        if freeze_batchnorm:
            for mname, module in self.named_modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    if hasattr(module, 'weight'):
                        module.weight.requires_grad_(False)
                    if hasattr(module, 'bias'):
                        module.bias.requires_grad_(False)
                    module.eval()
                    module.track_running_stats = False  # prevent running stats from updated during finetuning

                    logging.info(f"Froze module {mname}: {module}")

        lora_names = set([])
        for module in self.modules():  # access PT subclass method via inheritance
            if hasattr(module, 'lora_layer') and module.is_lora_available():
                for name, config in self.lora_cfg.items():
                    # Skip global lora config
                    if name == self.lora_global_cfg_key:
                        continue

                    # Check if lora is enabled or not
                    if self.lora_cfg[name]['enabled'] and name in module.lora_layer:
                        # Recursively set training mode of submodules
                        module.lora_layer[name].train()

                        # Recursively set grad required for submodules
                        for pname, param in module.lora_layer[name].named_parameters():
                            param.requires_grad_(True)

                        # unfreeze batch norm if any in the lora submodules
                        for mname, module_ in module.lora_layer[name].named_modules():
                            if isinstance(module_, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                                module_.track_running_stats = (
                                    True  # prevent running stats from updated during finetuning
                                )
                                logging.info(f"Unfroze lora module {mname}: {module_}")

                        lora_names.add(name)

        for name in lora_names:
            logging.info(f"Unfrozen lora : {name}")

    def forward_enabled_loras(self, input: 'torch.Tensor'):
        """
        Forward's all active loras one by one with the provided input, and chaining the outputs of each
        lora layer to the next.

        Utilizes the implicit merge strategy of each lora when computing the lora's output, and
        how that output will be merged back with the original input.

        **Note**:

        Args:
            input: The output tensor of the calling module is the input to the first lora, whose output
                is then chained to the next lora until all loras are consumed.

        Returns:
            The result tensor, after all active loras have finished their forward passes.
        """
        enabled_loras = self.get_enabled_loras()
        for lora_name in enabled_loras:
            lora_module = self.lora_layer[lora_name]

            if hasattr(lora_module, 'adapter_strategy'):
                strategy = (
                    lora_module.adapter_strategy
                )  # type: 'nemo.core.classes.mixins.adapter_mixin_strategies.AbstractLoraStrategy'
            else:
                raise AttributeError(
                    f"Lora module `{lora_name}` does not set the value `lora_strategy` ! "
                    f"Please set the value of the lora's strategy with the class "
                    f"{lora_module.__class__.__module}.{lora_module.__class__.__name__}."
                )

            # Call a single lora's forward, and accept its output as the new input for the next lora.
            input = self.forward_single_enabled_lora_(
                input, lora_module, lora_name=lora_name, lora_strategy=strategy
            )

        return input

    # Utility methods

    def resolve_lora_module_name_(self, name: str) -> (str, str):
        """
        Utility method to resolve a given global/module lora name to its components.
        Always returns a tuple representing (module_name, lora_name). ":" is used as the
        delimiter for denoting the module name vs the lora name.

        Will attempt to also resolve a given lora_name alone back to (module_name, lora_name)
        if the metadata config exists for access.

        Args:
            name: A global lora, or a module lora name (with structure module_name:lora_name).

        Returns:
            A tuple representing (module_name, lora_name). If a global lora is provided,
            module_name is set to ''.
        """
        # Attempt to split into module lora name, iff : exists in the given name.
        if ':' in name:
            splits = name.split(":")
            module_name = splits[0]
            lora_name = ":".join(splits[1:])
            return (module_name, lora_name)
        else:
            # Prepare default module name
            module_name = ''

            # Can be following cases:
            # 1) Loras are being restored. In this case, we need to resolve the module name from the config
            if hasattr(self, 'lora_cfg') and self.lora_cfg is not None:
                cfg = self.lora_cfg.get(self.lora_global_cfg_key, {})
                cfg = cfg.get(self.lora_metadata_cfg_key, {})
                cfg = cfg.get('modules', {})

                # Try to get the module for the given lora name, if available, else use default.
                module_name = cfg.get(name, '')

            # If the above cases dont hold, no module name provided when the user is adding a new lora.
            # Just return whatever module name was resolved, or the default
            return (module_name, name)

    def forward_single_enabled_lora_(
        self,
        input: torch.Tensor,
        lora_module: torch.nn.Module,
        *,
        lora_name: str,
        lora_strategy: 'nemo.core.classes.mixins.adapter_mixin_strategies.AbstractLoraStrategy',
    ):
        """
        Perform the forward step of a single lora module on some input data.

        **Note**: Subclasses can override this method to accommodate more complicate lora forward steps.

        Args:
            input: input: The output tensor of the calling module is the input to the first lora, whose output
                is then chained to the next lora until all loras are consumed.
            lora_module: The lora module that is currently required to perform the forward pass.
            lora_name: The resolved name of the lora that is undergoing the current forward pass.
            lora_strategy: A subclass of `AbstractLoraStrategy`, that determines how the
                output of the lora should be merged with the input, or if it should be merged at all.

        Returns:
            The result tensor, after the current active lora has finished its forward pass.
        """
        # (input: torch.Tensor, lora: torch.nn.Module, *, module: 'LoraModuleMixin')
        output = lora_strategy(input, lora_module, module=self)
        return output


class LoraModelPTMixin(LoraModuleMixin):
    """ Lora Mixin that can augment a ModelPT subclass with Lora support.

    This mixin class should be used only with a top level ModelPT subclass.
    This mixin class adds several utility methods which should be subclassed and overriden to
    propagated to the submodules as necessary.

    An Lora module is any Pytorch nn.Module that possess a few properties :

    - It's input and output dimension are the same, while the hidden dimension need not be the same.
    - The final layer of the Lora module is zero-initialized, so that the residual connection to the lora
        yields the original output.

    This mixin adds the following instance variables to the class this inherits it:

        -   `lora_layer`: A torch.nn.ModuleDict(), whose keys are the names of the lora (globally unique),
                and values are the Lora nn.Module().
        -   `lora_cfg`: A OmegaConf DictConfig object that holds the config of the loras that are initialized.
        -   `lora_global_cfg_key`: A str representing a key in the model config that can be provided by the user.
            The value resolves to `global_cfg`, and can be overridden via `model.cfg.loras.global_cfg.*`.

    .. note::

        This module **is** responsible for maintaining its config. At the ModelPT level, it will access and
        write Lora config information to `self.cfg.loras`.
    """

    def setup_loras(self):
        """
        Utility method that is called in the ASR ModelPT-implementation constructor, so as to restore any
        loras that were previously added.

        Should be overriden by the subclass for additional setup steps as required.

        This method should be called just once at constructor time.
        """
        # Test if `loras` is part of the config (injected from previous Lora additions)
        if 'loras' in self.cfg:
            # Set the global config of loras
            self.update_lora_cfg(self.cfg.loras)

            # Dispatch the call to the encoder, for every lora contained in the config.
            for lora_name, lora_cfg in self.cfg.loras.items():
                # reserve special key `model.loras.cfg`
                if lora_name == self.lora_global_cfg_key:
                    continue

                # Add the loras back to the model during setup
                # Add a guard so that during restoration, unique name check is disabled
                self._restoring_loras = True

                # Restore the unique lora
                self.add_lora(name=lora_name, cfg=lora_cfg)

                # Remove restoration guard
                del self._restoring_loras

                # Log the setup lora name
                module_name, lora_name = self.resolve_lora_module_name_(lora_name)

                if module_name != '':
                    full_lora_name = f'{module_name}:{lora_name}'
                else:
                    full_lora_name = lora_name

                logging.info(
                    f"Finished setup of lora : '{full_lora_name}'. Enabled: {lora_cfg.get('enabled', True)}."
                )

    def add_lora(self, name: str, cfg: DictConfig):
        """
        Add an Lora module to this model.

        Should be overridden by subclass and super() call must be used - this will setup the config.
        After calling super(), forward this call to modules that implement the mixin.

        Args:
            name: A globally unique name for the lora. Will be used to access, enable and disable loras.
            cfg: A DictConfig that contains at the bare minimum `__target__` to instantiate a new Lora module.
        """
        # Convert to DictConfig from dict or Dataclass
        if is_dataclass(cfg):
            cfg = OmegaConf.structured(cfg)

        if not isinstance(cfg, DictConfig):
            cfg = DictConfig(cfg)

        # Resolve the module name and lora name (if provided for the first time)
        module_name, lora_name = self.resolve_lora_module_name_(name)

        # Update the model.cfg with information about the new lora from cfg
        with open_dict(cfg), open_dict(self.cfg):
            # Construct the minimum config required to be updated by lora implementations
            if 'loras' not in self.cfg:
                self.cfg.loras = OmegaConf.create({})

            self.cfg.loras = _prepare_default_lora_config(
                global_key=self.lora_global_cfg_key, meta_key=self.lora_metadata_cfg_key, cfg=self.cfg.loras,
            )

            # If the lora is not being restored, force unique name to be provided for all loras.
            if hasattr(self, '_restoring_loras') and self._restoring_loras is not True:
                if lora_name in self.cfg.loras:
                    raise ValueError(f"Attempting to add multiple loras with the same name ({lora_name}) !")

            # Inject the module name in the lora metadata cfg
            gcfg = self.lora_global_cfg_key
            mcfg = self.lora_metadata_cfg_key
            self.cfg.loras[gcfg][mcfg]['modules'][lora_name] = module_name

            # By default, enable the lora that is being added
            if 'enabled' not in cfg:
                cfg['enabled'] = True

            # Assign the
            self.cfg.loras[lora_name] = OmegaConf.create(cfg)

            # Set the global config of loras
            self.update_lora_cfg(self.cfg.loras)

            self.check_valid_model_with_lora_support_()

    def is_lora_available(self) -> bool:
        """
        Checks if any Lora module has been instantiated.

        Should be overridden by the subclass.

        Returns:
            bool, determining if any Lora module has been instantiated. Returns true even if the loras are
            enabled or disabled, false only if no loras exist.
        """
        self.check_valid_model_with_lora_support_()

        if 'loras' in self.cfg:
            self.update_lora_cfg(self.cfg.loras)

        return 'loras' in self.cfg and len(self.get_enabled_loras()) > 0

    def set_enabled_loras(self, name: Optional[str] = None, enabled: bool = True):
        """
        Updated the internal lora config, determining if an lora (or all loras) are either
        enabled or disabled.

        A common user pattern would be to disable all loras (either after adding them, or restoring a model
        with pre-existing loras) and then simply enable one of the loras.

        Should be overridden by subclass and super() call must be used - this will setup the config.
        After calling super(), forward this call to modules that implement the mixin.

        .. code::

            model.set_enabled_loras(enabled=False)
            model.set_enabled_loras(name=<some lora name>, enabled=True)

        Args:
            name: Optional str. If a str name is given, the config will be updated to the value of `enabled`.
                If no name is given, then all loras will be enabled/disabled.
            enabled: Bool, determines if the lora(s) will be enabled/disabled.
        """
        self.check_valid_model_with_lora_support_()

        # Update the lora config with information about whether it is enabled/disabled.
        with open_dict(self.cfg.loras):
            # If no name is provided, update all loras.
            if name is None:
                for key in self.cfg.loras.keys():
                    # Skip the global lora config
                    if key == self.lora_global_cfg_key:
                        continue

                    self.cfg.loras[key]['enabled'] = enabled
                    logging.info(f"Setting lora '{key}' status : Enabled = {enabled}")

            else:
                # Resolve the module name and lora name
                module_name, lora_name = self.resolve_lora_module_name_(name)

                # Cannot set the state of the global config for loras
                if lora_name == self.lora_global_cfg_key:
                    raise ValueError(
                        f'Cannot set the state of the global config of loras, '
                        f'given name = `{self.lora_global_cfg_key}`'
                    )

                # Otherwise, update just the specified lora.
                self.cfg.loras[lora_name]['enabled'] = enabled
                logging.info(f"Setting lora '{name}' status : Enabled = {enabled}")

            self.update_lora_cfg(self.cfg.loras)

    def get_enabled_loras(self) -> List[str]:
        """
        Returns a list of all enabled loras.

        Should be implemented by the subclass.

        Returns:
            A list of str names of each enabled lora(s).
        """
        self.check_valid_model_with_lora_support_()

        if 'loras' in self.cfg:
            self.update_lora_cfg(self.cfg.loras)
        return []

    def check_valid_model_with_lora_support_(self):
        """
        Utility method to test if the subclass of this mixin is an appropriate subclass of ModelPT itself.

        Should be implemented by the subclass.
        """
        pass

    def save_loras(self, filepath: str, name: str = None):
        """
        Utility method that saves only the lora module(s), and not the entire model itself.
        This allows the sharing of loras which are often just a fraction of the size of the full model,
        enabling easier deliver.

        Note: The saved file is a pytorch compatible pickle file, containing the state dicts of the lora(s),
            as well as a binary representation of the lora config.

        Args:
            filepath: A str filepath where the .pt file that will contain the lora state dict.
            name: Optional name of the lora that will be saved to this file. If None is passed,
                all loras will be saved to the file. The name can be either the global name (lora_name),
                or the module level name (module:lora_name).
        """
        if not hasattr(self, 'cfg') or 'loras' not in self.cfg:
            raise AttributeError("No loras have been added to this model, so no loras can be saved.")

        output_dict = {}

        # Normalize the name to a list of strings
        if isinstance(name, str):
            name = [name]

        if name is None:
            name = self.cfg.loras.keys()

        # Assert that the config must be present to save and restore the loras.
        if not hasattr(self.cfg, 'loras'):
            raise ValueError(
                "The model has no lora config, therefore it cannot save any lora. "
                "Please first add one or more loras to generate the config."
            )

        # For each lora name (either global lora or module loras)
        for lora_name in name:
            if lora_name != self.lora_global_cfg_key:
                # Resolve the lora name into its components
                module_name, lora_name = self.resolve_lora_module_name_(lora_name)

                # Reconstruct a module lora's original name. For global loras, the '' is preserved.
                if module_name == '':
                    key = lora_name
                else:
                    key = f'{module_name}:{lora_name}'
                output_dict[key] = []

                # Search all modules with the following criterion -
                # It must be an implementation of LoraModuleMixin.
                # It must have the attribute `lora_name`.
                # It must match the lora name provided by the user.
                for module in self.modules():
                    if (
                        isinstance(module, LoraModuleMixin)
                        and hasattr(module, 'lora_name')
                        and module.lora_name == lora_name
                    ):
                        # If all match, extract the state dict into a list of state dicts.
                        # This is because one name can be shared within one model by multiple loras bearing
                        # a common name. This can occur when the lora is common to a module which has multiple
                        # layers and blocks, all of which require an lora.
                        state_dict = module.state_dict()
                        output_dict[key].append(state_dict)

        # Preserve the binary OmegaConf dictionary of the model's lora config
        output_dict['__cfg__'] = self.cfg.loras

        # Finally, save the lora state dict(s).
        torch.save(output_dict, filepath)

    def load_loras(self, filepath: str, name: str = None, map_location: str = None, strict: bool = True):
        """
        Utility method that restores only the lora module(s), and not the entire model itself.
        This allows the sharing of loras which are often just a fraction of the size of the full model,
        enabling easier deliver.

        Note: During restoration, assumes that the model does not currently already have an lora with
            the name (if provided), or any lora that shares a name with the state dict's modules
            (if name is not provided). This is to ensure that each lora name is globally unique
            in a model.

        Args:
            filepath: Filepath of the .pt file.
            name: Optional name of the lora that will be saved to this file. If None is passed,
                all loras will be saved to the file. The name must be either the global name (lora_name),
                or the module level name (module:lora_name), whichever exactly matches the state dict.
            map_location: Pytorch flag, where to place the lora(s) state dict(s).
            strict: Pytorch flag, whether to load the weights of the lora(s) strictly or not.
        """
        # Determine device
        if map_location is None:
            if torch.cuda.is_available():
                map_location = 'cuda'
            else:
                map_location = 'cpu'

        # Load the state dict and extract the internal config
        state_dict = torch.load(filepath, map_location=map_location)
        config = state_dict.pop('__cfg__')

        # Normalize the name to a list of names (exact match with the state dict)
        if isinstance(name, str):
            name = [name]

        if name is None:
            name = list(config.keys())

        # For all module:lora names (note, for global modules, we ignore the module: part)
        for module_lora_name in name:
            # Extract current config as copy
            internal_lora_cfg = None
            if hasattr(self, 'lora_cfg') and self.lora_cfg is not None:
                internal_lora_cfg = self.lora_cfg

            # Override internal lora config with restoration config
            self.lora_cfg = config

            # Resolve the lora name and extract the lora's config from the checkpoint.
            module_name, lora_name = self.resolve_lora_module_name_(module_lora_name)
            lora_cfg = config[lora_name]

            # Recreate the module:lora_name
            if module_name == '':
                module_lora_name = lora_name
            else:
                module_lora_name = f'{module_name}:{lora_name}'

            # Reset internal lora config
            self.lora_cfg = internal_lora_cfg

            # Skip the global config key
            if lora_name == self.lora_global_cfg_key:
                continue

            # Restore weights with exact key, if it fails, give useful error message.
            try:
                lora_state = state_dict[module_lora_name]
            except KeyError:
                all_keys = list(state_dict.keys())
                raise KeyError(
                    f"Requested to load lora with name `{module_lora_name}`, but could not "
                    f"the lora in the state dict. \nAvailable lora names in state dict are: "
                    f"{all_keys}"
                )

            # If key was found, add a new lora with random weights
            self.add_lora(name=module_lora_name, cfg=lora_cfg)

            # Determine apriori how many modules must be loaded from the state dict
            # This is dont to guarentee that partial match does not occur, only exact match
            # between state dict and the loras parameters will be allowed.
            modules_to_load = []  # type: List[torch.nn.Module]
            for module in self.modules():
                if (
                    isinstance(module, LoraModuleMixin)
                    and hasattr(module, 'lora_name')
                    and module.lora_name == lora_name
                ):
                    modules_to_load.append(module)

            # Assert that the number of states in the state dict matches the newly created lora
            if len(lora_state) != len(modules_to_load):
                raise ValueError(
                    f"The number of loras in current model ({len(modules_to_load)}) does not "
                    f"match the number of modules in the state dict for lora `{lora_name}`: "
                    f"({len(lora_state)})"
                )

            # For the pair of (lora_state_in_checkpoint, lora_in_model), restore the weights
            for state, module in zip(lora_state, modules_to_load):
                module.load_state_dict(state, strict=strict)

            # delete the dictionaries to preserve memory for next lora
            del lora_state, modules_to_load

    def update_lora_cfg(self, cfg: DictConfig):
        """
        Utility method to recursively update all of the Lora module configs with the provided config.

        .. note::

            It is not a (deep)copy, but a reference copy. Changes made to the config will be reflected to
            lora submodules, but it is still encouraged to explicitly update the lora_cfg using this method.

        Args:
            cfg: DictConfig containing the value of `model.cfg.loras`.
        """
        for module in self.modules():  # access PT subclass method via inheritance
            if isinstance(module, LoraModuleMixin):
                module.lora_cfg = cfg

    @property
    def lora_module_names(self) -> List[str]:
        """
        List of valid lora modules that are supported by the model.

        **Note**: Subclasses should override this property and return a list of str names, of all the modules
            that they support, which will enable users to determine where to place the lora modules.

        Returns:
            A list of str, one for each of the lora modules that are supported. By default, the subclass
            should support the "global lora" ('').
        """
        return ['']
