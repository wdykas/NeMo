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

# Global registry of all prefixs
PREFIX_REGISTRY = {}


@dataclass
class PrefixRegistryInfo:
    base_class: type
    prefix_class: type

    # generated automatically
    base_class_path: str = ""
    prefix_class_path: str = ""

    def __post_init__(self):
        self.base_class_path = f'{self.base_class.__module__}.{self.base_class.__name__}'
        self.prefix_class_path = f'{self.prefix_class.__module__}.{self.prefix_class.__name__}'


def register_prefix(base_class: type, prefix_class: type):
    """
    Registers a pair (Base class, Prefix class) into the prefix registry, used for de-referencing.

    Args:
        base_class: A Class, which is the base class of the object.
        prefix_class: A Class, which is the subclass of the base class, and implements the Prefix mixin methods.
    """
    global PREFIX_REGISTRY
    base_class_path = f'{base_class.__module__}.{base_class.__name__}'
    prefix_class_path = f'{prefix_class.__module__}.{prefix_class.__name__}'

    # test if base class already in registry
    if base_class_path in PREFIX_REGISTRY:
        raise ValueError(f"`{base_class_path}` has already been added to the prefix registry !")

    # test if prefix is a subclass of the base class
    if not issubclass(prefix_class, base_class):
        raise ValueError(f"`{prefix_class_path}` is not a sub-class of {base_class_path} !")

    # register the base class : prefix class pair
    PREFIX_REGISTRY[base_class_path] = PrefixRegistryInfo(base_class=base_class, prefix_class=prefix_class)

    # attach prefix class to base class
    base_class._meta_prefix_class = prefix_class

    # attach base class to prefix class
    prefix_class._meta_base_class = base_class


def get_registered_prefix(cls: Union[str, type]) -> Optional[PrefixRegistryInfo]:
    """
    Resolves a provided `cls` (whether str path to class, a registered base or an prefix class)
    to obtain the metadata for the prefix.

    Args:
        cls: Can be a str (absolute path to a class), a base class or an prefix class (which have already
            been registered).

    Returns:
        A PrefixRegistryInfo object if it could resolve successfully, otherwise None.
    """
    global PREFIX_REGISTRY
    if isinstance(cls, str):
        cls = model_utils.import_class_by_path(cls)

    # If an prefix class was provided, de-reference its base class
    if hasattr(cls, '_meta_base_class'):
        cls = cls._meta_base_class

    class_path = f'{cls.__module__}.{cls.__name__}'

    # If base class, check registry
    if class_path in PREFIX_REGISTRY:
        return PREFIX_REGISTRY[class_path]

    return None


def _prepare_default_prefix_config(*, global_key: str, meta_key: str, cfg: DictConfig = None) -> DictConfig:
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


class PrefixModuleMixin(ABC):
    """ Generic Prefix Mixin that can augment any torch.nn.Module with Prefix module support.

    This mixin class adds a hierarchical way to add any type of Prefix modules to a pre-existing module.
    Since Models are inherently also nn.Module, this mixin can be attached to any Model or Module.
    This mixin class adds several utility methods which are utilized or overridden as necessary.

    An Prefix module is any Pytorch nn.Module that possess a few properties :

        -   It's input and output dimension are the same, while the hidden dimension need not be the same.
        -   The final layer of the Prefix module is zero-initialized, so that the residual connection to the prefix
                yields the original output.

    This mixin adds the following instance variables to the class this inherits it:

        -   `prefix_layer`: A torch.nn.ModuleDict(), whose keys are the names of the prefix (globally unique),
                and values are the Prefix nn.Module().
        -   `prefix_cfg`: A OmegaConf DictConfig object that holds the config of the prefixs that are initialized.
        -   `prefix_name`: A str resolved name which is unique key globally, but more than one modules may share
                this name.
        -   `prefix_global_cfg_key`: A str representing a key in the model config that can be provided by the user.
                The value resolves to `global_cfg`, and can be overridden via `model.cfg.prefixs.global_cfg.*`.
        -   `prefix_metadata_cfg_key`: A str representing a key in the model config that is used to preserve the
                metadata of the prefix config.

    **Note**: This module is **not** responsible for maintaining its config. Subclasses must ensure config is updated
        or preserved as needed. It is the responsibility of the subclasses to propagate the most up to date config to
        lower layers.
    """

    prefix_global_cfg_key = "global_cfg"
    prefix_metadata_cfg_key = "prefix_meta_cfg"

    def add_prefix(self, name: str, cfg: DictConfig):
        """
        Add an Prefix module to this module.

        Args:
            name: A globally unique name for the prefix. Will be used to access, enable and disable prefixs.
            cfg: A DictConfig or Dataclass that contains at the bare minimum `__target__` to instantiate a
                new Prefix module.
        """
        # Convert to DictConfig from dict or Dataclass
        if is_dataclass(cfg):
            cfg = OmegaConf.structured(cfg)

        if not isinstance(cfg, DictConfig):
            cfg = DictConfig(cfg)

        # Add prefix_layer ModuleDict() if not present.
        if not hasattr(self, 'prefix_layer'):
            self.prefix_layer = nn.ModuleDict()

        # Add prefix_cfg if it doesnt exist or hasnt been assigned yet.
        if not hasattr(self, 'prefix_cfg'):
            self.prefix_cfg = OmegaConf.create({})

        # Resolve the module name and prefix name (if module name is provided)
        _, prefix_name = self.resolve_prefix_module_name_(name)

        # Add prefix_name to this module for later identification
        self.prefix_name = prefix_name

        # Assert that name is globally unique to all prefixs.
        if prefix_name in self.prefix_layer:
            raise ValueError(
                f"Prefix with name `{name}` already exists ! Prefix names = {list(self.prefix_layer.keys())}"
            )

        # Assert that name is not `prefix_global_cfg_key`
        if prefix_name == self.prefix_global_cfg_key:
            raise ValueError(f"Prefixs cannot have the reserved name : `{self.prefix_global_cfg_key}`")

        # Update internal config and instantiate the Prefix module
        with open_dict(cfg), open_dict(self.prefix_cfg):
            prefix_enabled = cfg.pop('enabled', True)
            self.prefix_layer[prefix_name] = instantiate(cfg)

            cfg['enabled'] = prefix_enabled
            self.prefix_cfg[prefix_name] = cfg

    def is_prefix_available(self) -> bool:
        """
        Checks if any Prefix module has been instantiated.

        Returns:
            bool, determining if any Prefix module has been instantiated. Returns true even if the prefixs are
            enabled or disabled, false only if no prefixs exist.
        """
        if hasattr(self, 'prefix_layer'):
            return self.prefix_layer is not None and len(self.prefix_layer) > 0
        return False

    def set_enabled_prefixs(self, name: Optional[str] = None, enabled: bool = True):
        """
        Updated the internal prefix config, determining if an prefix (or all prefixs) are either
        enabled or disabled.

        A common user pattern would be to disable all prefixs (either after adding them, or restoring a model
        with pre-existing prefixs) and then simply enable one of the prefixs.

        .. code::

            module.set_enabled_prefixs(enabled=False)
            module.set_enabled_prefixs(name=<some prefix name>, enabled=True)

        Args:
            name: Optional str. If a str name is given, the config will be updated to the value of `enabled`.
                If no name is given, then all prefixs will be enabled/disabled.
            enabled: Bool, determines if the prefix(s) will be enabled/disabled.
        """
        if not self.is_prefix_available():
            raise ValueError("No prefix is available to enable/disable")

        # If name is None, enable/disable all prefixs.
        if name is None:
            for key, config in self.prefix_cfg.items():
                # Skip the global prefix config
                if key == self.prefix_global_cfg_key:
                    continue

                # Enable/Disable the current prefix
                self.prefix_cfg[key]['enabled'] = enabled
        else:
            _, prefix_name = self.resolve_prefix_module_name_(name)

            # Cannot set the state of the global config for prefixs
            if prefix_name == self.prefix_global_cfg_key:
                raise ValueError(
                    f'Cannot set the state of the global config of prefixs, '
                    f'given name = `{self.prefix_global_cfg_key}`'
                )

            # Enable/Disable just named prefix
            self.prefix_cfg[prefix_name]['enabled'] = enabled

    def get_enabled_prefixs(self) -> List[str]:
        """
        Returns a list of all enabled prefixs names. The names will always be the resolved names, without
        module info.

        Returns:
            A list of str names of each enabled prefix names(s).
        """
        if not self.is_prefix_available():
            return []

        # populate set of available modules (by name)
        available_module_names = set([])
        if hasattr(self, 'prefix_layer'):
            available_module_names.update(list(self.prefix_layer.keys()))

        enabled_prefixs = []
        for name, config in self.prefix_cfg.items():
            # Skip the global prefix config
            if name == self.prefix_global_cfg_key:
                continue

            # If name is in the current available modules, and it is enabled in the config
            if name in available_module_names and self.prefix_cfg[name]['enabled']:
                enabled_prefixs.append(name)

        return enabled_prefixs

    # Inherited methods that dont need to be overridden

    def unfreeze_enabled_prefixs(self, freeze_batchnorm: bool = True) -> None:
        """
        Utility method to unfreeze only the enabled Prefix module(s).

        A common user pattern is to freeze all the modules (including all the prefixs), and then
        unfreeze just the required prefixs.

        .. code::

            module.freeze()  # only available to nemo.core.NeuralModule !
            module.unfreeze_enabled_prefixs()

        Args:
            freeze_batchnorm: An optional (and recommended) practice of freezing the updates to the moving average
                buffers of any and all BatchNorm*D layers. This is necessary to ensure that disabling all prefixs
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

        prefix_names = set([])
        for module in self.modules():  # access PT subclass method via inheritance
            if hasattr(module, 'prefix_layer') and module.is_prefix_available():
                for name, config in self.prefix_cfg.items():
                    # Skip global prefix config
                    if name == self.prefix_global_cfg_key:
                        continue

                    # Check if prefix is enabled or not
                    if self.prefix_cfg[name]['enabled'] and name in module.prefix_layer:
                        # Recursively set training mode of submodules
                        module.prefix_layer[name].train()

                        # Recursively set grad required for submodules
                        for pname, param in module.prefix_layer[name].named_parameters():
                            param.requires_grad_(True)

                        # unfreeze batch norm if any in the prefix submodules
                        for mname, module_ in module.prefix_layer[name].named_modules():
                            if isinstance(module_, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                                module_.track_running_stats = (
                                    True  # prevent running stats from updated during finetuning
                                )
                                logging.info(f"Unfroze prefix module {mname}: {module_}")

                        prefix_names.add(name)

        for name in prefix_names:
            logging.info(f"Unfrozen prefix : {name}")

    def forward_enabled_prefixs(self, input: 'torch.Tensor'):
        """
        Forward's all active prefixs one by one with the provided input, and chaining the outputs of each
        prefix layer to the next.

        Utilizes the implicit merge strategy of each prefix when computing the prefix's output, and
        how that output will be merged back with the original input.

        **Note**:

        Args:
            input: The output tensor of the calling module is the input to the first prefix, whose output
                is then chained to the next prefix until all prefixs are consumed.

        Returns:
            The result tensor, after all active prefixs have finished their forward passes.
        """
        enabled_prefixs = self.get_enabled_prefixs()
        for prefix_name in enabled_prefixs:
            prefix_module = self.prefix_layer[prefix_name]

            if hasattr(prefix_module, 'adapter_strategy'):
                strategy = (
                    prefix_module.adapter_strategy
                )  # type: 'nemo.core.classes.mixins.adapter_mixin_strategies.AbstractPrefixStrategy'
            else:
                raise AttributeError(
                    f"Prefix module `{prefix_name}` does not set the value `prefix_strategy` ! "
                    f"Please set the value of the prefix's strategy with the class "
                    f"{prefix_module.__class__.__module}.{prefix_module.__class__.__name__}."
                )

            # Call a single prefix's forward, and accept its output as the new input for the next prefix.
            input = self.forward_single_enabled_prefix_(
                input, prefix_module, prefix_name=prefix_name, prefix_strategy=strategy
            )

        return input

    # Utility methods

    def resolve_prefix_module_name_(self, name: str) -> (str, str):
        """
        Utility method to resolve a given global/module prefix name to its components.
        Always returns a tuple representing (module_name, prefix_name). ":" is used as the
        delimiter for denoting the module name vs the prefix name.

        Will attempt to also resolve a given prefix_name alone back to (module_name, prefix_name)
        if the metadata config exists for access.

        Args:
            name: A global prefix, or a module prefix name (with structure module_name:prefix_name).

        Returns:
            A tuple representing (module_name, prefix_name). If a global prefix is provided,
            module_name is set to ''.
        """
        # Attempt to split into module prefix name, iff : exists in the given name.
        if ':' in name:
            splits = name.split(":")
            module_name = splits[0]
            prefix_name = ":".join(splits[1:])
            return (module_name, prefix_name)
        else:
            # Prepare default module name
            module_name = ''

            # Can be following cases:
            # 1) Prefixs are being restored. In this case, we need to resolve the module name from the config
            if hasattr(self, 'prefix_cfg') and self.prefix_cfg is not None:
                cfg = self.prefix_cfg.get(self.prefix_global_cfg_key, {})
                cfg = cfg.get(self.prefix_metadata_cfg_key, {})
                cfg = cfg.get('modules', {})

                # Try to get the module for the given prefix name, if available, else use default.
                module_name = cfg.get(name, '')

            # If the above cases dont hold, no module name provided when the user is adding a new prefix.
            # Just return whatever module name was resolved, or the default
            return (module_name, name)

    def forward_single_enabled_prefix_(
        self,
        input: torch.Tensor,
        prefix_module: torch.nn.Module,
        *,
        prefix_name: str,
        prefix_strategy: 'nemo.core.classes.mixins.adapter_mixin_strategies.AbstractPrefixStrategy',
    ):
        """
        Perform the forward step of a single prefix module on some input data.

        **Note**: Subclasses can override this method to accommodate more complicate prefix forward steps.

        Args:
            input: input: The output tensor of the calling module is the input to the first prefix, whose output
                is then chained to the next prefix until all prefixs are consumed.
            prefix_module: The prefix module that is currently required to perform the forward pass.
            prefix_name: The resolved name of the prefix that is undergoing the current forward pass.
            prefix_strategy: A subclass of `AbstractPrefixStrategy`, that determines how the
                output of the prefix should be merged with the input, or if it should be merged at all.

        Returns:
            The result tensor, after the current active prefix has finished its forward pass.
        """
        # (input: torch.Tensor, prefix: torch.nn.Module, *, module: 'PrefixModuleMixin')
        output = prefix_strategy(input, prefix_module, module=self)
        return output


class PrefixModelPTMixin(PrefixModuleMixin):
    """ Prefix Mixin that can augment a ModelPT subclass with Prefix support.

    This mixin class should be used only with a top level ModelPT subclass.
    This mixin class adds several utility methods which should be subclassed and overriden to
    propagated to the submodules as necessary.

    An Prefix module is any Pytorch nn.Module that possess a few properties :

    - It's input and output dimension are the same, while the hidden dimension need not be the same.
    - The final layer of the Prefix module is zero-initialized, so that the residual connection to the prefix
        yields the original output.

    This mixin adds the following instance variables to the class this inherits it:

        -   `prefix_layer`: A torch.nn.ModuleDict(), whose keys are the names of the prefix (globally unique),
                and values are the Prefix nn.Module().
        -   `prefix_cfg`: A OmegaConf DictConfig object that holds the config of the prefixs that are initialized.
        -   `prefix_global_cfg_key`: A str representing a key in the model config that can be provided by the user.
            The value resolves to `global_cfg`, and can be overridden via `model.cfg.prefixs.global_cfg.*`.

    .. note::

        This module **is** responsible for maintaining its config. At the ModelPT level, it will access and
        write Prefix config information to `self.cfg.prefixs`.
    """

    def setup_prefixs(self):
        """
        Utility method that is called in the ASR ModelPT-implementation constructor, so as to restore any
        prefixs that were previously added.

        Should be overriden by the subclass for additional setup steps as required.

        This method should be called just once at constructor time.
        """
        # Test if `prefixs` is part of the config (injected from previous Prefix additions)
        if 'prefixs' in self.cfg:
            # Set the global config of prefixs
            self.update_prefix_cfg(self.cfg.prefixs)

            # Dispatch the call to the encoder, for every prefix contained in the config.
            for prefix_name, prefix_cfg in self.cfg.prefixs.items():
                # reserve special key `model.prefixs.cfg`
                if prefix_name == self.prefix_global_cfg_key:
                    continue

                # Add the prefixs back to the model during setup
                # Add a guard so that during restoration, unique name check is disabled
                self._restoring_prefixs = True

                # Restore the unique prefix
                self.add_prefix(name=prefix_name, cfg=prefix_cfg)

                # Remove restoration guard
                del self._restoring_prefixs

                # Log the setup prefix name
                module_name, prefix_name = self.resolve_prefix_module_name_(prefix_name)

                if module_name != '':
                    full_prefix_name = f'{module_name}:{prefix_name}'
                else:
                    full_prefix_name = prefix_name

                logging.info(
                    f"Finished setup of prefix : '{full_prefix_name}'. Enabled: {prefix_cfg.get('enabled', True)}."
                )

    def add_prefix(self, name: str, cfg: DictConfig):
        """
        Add an Prefix module to this model.

        Should be overridden by subclass and super() call must be used - this will setup the config.
        After calling super(), forward this call to modules that implement the mixin.

        Args:
            name: A globally unique name for the prefix. Will be used to access, enable and disable prefixs.
            cfg: A DictConfig that contains at the bare minimum `__target__` to instantiate a new Prefix module.
        """
        # Convert to DictConfig from dict or Dataclass
        if is_dataclass(cfg):
            cfg = OmegaConf.structured(cfg)

        if not isinstance(cfg, DictConfig):
            cfg = DictConfig(cfg)

        # Resolve the module name and prefix name (if provided for the first time)
        module_name, prefix_name = self.resolve_prefix_module_name_(name)

        # Update the model.cfg with information about the new prefix from cfg
        with open_dict(cfg), open_dict(self.cfg):
            # Construct the minimum config required to be updated by prefix implementations
            if 'prefixs' not in self.cfg:
                self.cfg.prefixs = OmegaConf.create({})

            self.cfg.prefixs = _prepare_default_prefix_config(
                global_key=self.prefix_global_cfg_key, meta_key=self.prefix_metadata_cfg_key, cfg=self.cfg.prefixs,
            )

            # If the prefix is not being restored, force unique name to be provided for all prefixs.
            if hasattr(self, '_restoring_prefixs') and self._restoring_prefixs is not True:
                if prefix_name in self.cfg.prefixs:
                    raise ValueError(f"Attempting to add multiple prefixs with the same name ({prefix_name}) !")

            # Inject the module name in the prefix metadata cfg
            gcfg = self.prefix_global_cfg_key
            mcfg = self.prefix_metadata_cfg_key
            self.cfg.prefixs[gcfg][mcfg]['modules'][prefix_name] = module_name

            # By default, enable the prefix that is being added
            if 'enabled' not in cfg:
                cfg['enabled'] = True

            # Assign the
            self.cfg.prefixs[prefix_name] = OmegaConf.create(cfg)

            # Set the global config of prefixs
            self.update_prefix_cfg(self.cfg.prefixs)

            self.check_valid_model_with_prefix_support_()

    def is_prefix_available(self) -> bool:
        """
        Checks if any Prefix module has been instantiated.

        Should be overridden by the subclass.

        Returns:
            bool, determining if any Prefix module has been instantiated. Returns true even if the prefixs are
            enabled or disabled, false only if no prefixs exist.
        """
        self.check_valid_model_with_prefix_support_()

        if 'prefixs' in self.cfg:
            self.update_prefix_cfg(self.cfg.prefixs)

        return 'prefixs' in self.cfg and len(self.get_enabled_prefixs()) > 0

    def set_enabled_prefixs(self, name: Optional[str] = None, enabled: bool = True):
        """
        Updated the internal prefix config, determining if an prefix (or all prefixs) are either
        enabled or disabled.

        A common user pattern would be to disable all prefixs (either after adding them, or restoring a model
        with pre-existing prefixs) and then simply enable one of the prefixs.

        Should be overridden by subclass and super() call must be used - this will setup the config.
        After calling super(), forward this call to modules that implement the mixin.

        .. code::

            model.set_enabled_prefixs(enabled=False)
            model.set_enabled_prefixs(name=<some prefix name>, enabled=True)

        Args:
            name: Optional str. If a str name is given, the config will be updated to the value of `enabled`.
                If no name is given, then all prefixs will be enabled/disabled.
            enabled: Bool, determines if the prefix(s) will be enabled/disabled.
        """
        self.check_valid_model_with_prefix_support_()

        # Update the prefix config with information about whether it is enabled/disabled.
        with open_dict(self.cfg.prefixs):
            # If no name is provided, update all prefixs.
            if name is None:
                for key in self.cfg.prefixs.keys():
                    # Skip the global prefix config
                    if key == self.prefix_global_cfg_key:
                        continue

                    self.cfg.prefixs[key]['enabled'] = enabled
                    logging.info(f"Setting prefix '{key}' status : Enabled = {enabled}")

            else:
                # Resolve the module name and prefix name
                module_name, prefix_name = self.resolve_prefix_module_name_(name)

                # Cannot set the state of the global config for prefixs
                if prefix_name == self.prefix_global_cfg_key:
                    raise ValueError(
                        f'Cannot set the state of the global config of prefixs, '
                        f'given name = `{self.prefix_global_cfg_key}`'
                    )

                # Otherwise, update just the specified prefix.
                self.cfg.prefixs[prefix_name]['enabled'] = enabled
                logging.info(f"Setting prefix '{name}' status : Enabled = {enabled}")

            self.update_prefix_cfg(self.cfg.prefixs)

    def get_enabled_prefixs(self) -> List[str]:
        """
        Returns a list of all enabled prefixs.

        Should be implemented by the subclass.

        Returns:
            A list of str names of each enabled prefix(s).
        """
        self.check_valid_model_with_prefix_support_()

        if 'prefixs' in self.cfg:
            self.update_prefix_cfg(self.cfg.prefixs)
        return []

    def check_valid_model_with_prefix_support_(self):
        """
        Utility method to test if the subclass of this mixin is an appropriate subclass of ModelPT itself.

        Should be implemented by the subclass.
        """
        pass

    def save_prefixs(self, filepath: str, name: str = None):
        """
        Utility method that saves only the prefix module(s), and not the entire model itself.
        This allows the sharing of prefixs which are often just a fraction of the size of the full model,
        enabling easier deliver.

        Note: The saved file is a pytorch compatible pickle file, containing the state dicts of the prefix(s),
            as well as a binary representation of the prefix config.

        Args:
            filepath: A str filepath where the .pt file that will contain the prefix state dict.
            name: Optional name of the prefix that will be saved to this file. If None is passed,
                all prefixs will be saved to the file. The name can be either the global name (prefix_name),
                or the module level name (module:prefix_name).
        """
        if not hasattr(self, 'cfg') or 'prefixs' not in self.cfg:
            raise AttributeError("No prefixs have been added to this model, so no prefixs can be saved.")

        output_dict = {}

        # Normalize the name to a list of strings
        if isinstance(name, str):
            name = [name]

        if name is None:
            name = self.cfg.prefixs.keys()

        # Assert that the config must be present to save and restore the prefixs.
        if not hasattr(self.cfg, 'prefixs'):
            raise ValueError(
                "The model has no prefix config, therefore it cannot save any prefix. "
                "Please first add one or more prefixs to generate the config."
            )

        # For each prefix name (either global prefix or module prefixs)
        for prefix_name in name:
            if prefix_name != self.prefix_global_cfg_key:
                # Resolve the prefix name into its components
                module_name, prefix_name = self.resolve_prefix_module_name_(prefix_name)

                # Reconstruct a module prefix's original name. For global prefixs, the '' is preserved.
                if module_name == '':
                    key = prefix_name
                else:
                    key = f'{module_name}:{prefix_name}'
                output_dict[key] = []

                # Search all modules with the following criterion -
                # It must be an implementation of PrefixModuleMixin.
                # It must have the attribute `prefix_name`.
                # It must match the prefix name provided by the user.
                for module in self.modules():
                    if (
                        isinstance(module, PrefixModuleMixin)
                        and hasattr(module, 'prefix_name')
                        and module.prefix_name == prefix_name
                    ):
                        # If all match, extract the state dict into a list of state dicts.
                        # This is because one name can be shared within one model by multiple prefixs bearing
                        # a common name. This can occur when the prefix is common to a module which has multiple
                        # layers and blocks, all of which require an prefix.
                        state_dict = module.state_dict()
                        output_dict[key].append(state_dict)

        # Preserve the binary OmegaConf dictionary of the model's prefix config
        output_dict['__cfg__'] = self.cfg.prefixs

        # Finally, save the prefix state dict(s).
        torch.save(output_dict, filepath)

    def load_prefixs(self, filepath: str, name: str = None, map_location: str = None, strict: bool = True):
        """
        Utility method that restores only the prefix module(s), and not the entire model itself.
        This allows the sharing of prefixs which are often just a fraction of the size of the full model,
        enabling easier deliver.

        Note: During restoration, assumes that the model does not currently already have an prefix with
            the name (if provided), or any prefix that shares a name with the state dict's modules
            (if name is not provided). This is to ensure that each prefix name is globally unique
            in a model.

        Args:
            filepath: Filepath of the .pt file.
            name: Optional name of the prefix that will be saved to this file. If None is passed,
                all prefixs will be saved to the file. The name must be either the global name (prefix_name),
                or the module level name (module:prefix_name), whichever exactly matches the state dict.
            map_location: Pytorch flag, where to place the prefix(s) state dict(s).
            strict: Pytorch flag, whether to load the weights of the prefix(s) strictly or not.
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

        # For all module:prefix names (note, for global modules, we ignore the module: part)
        for module_prefix_name in name:
            # Extract current config as copy
            internal_prefix_cfg = None
            if hasattr(self, 'prefix_cfg') and self.prefix_cfg is not None:
                internal_prefix_cfg = self.prefix_cfg

            # Override internal prefix config with restoration config
            self.prefix_cfg = config

            # Resolve the prefix name and extract the prefix's config from the checkpoint.
            module_name, prefix_name = self.resolve_prefix_module_name_(module_prefix_name)
            prefix_cfg = config[prefix_name]

            # Recreate the module:prefix_name
            if module_name == '':
                module_prefix_name = prefix_name
            else:
                module_prefix_name = f'{module_name}:{prefix_name}'

            # Reset internal prefix config
            self.prefix_cfg = internal_prefix_cfg

            # Skip the global config key
            if prefix_name == self.prefix_global_cfg_key:
                continue

            # Restore weights with exact key, if it fails, give useful error message.
            try:
                prefix_state = state_dict[module_prefix_name]
            except KeyError:
                all_keys = list(state_dict.keys())
                raise KeyError(
                    f"Requested to load prefix with name `{module_prefix_name}`, but could not "
                    f"the prefix in the state dict. \nAvailable prefix names in state dict are: "
                    f"{all_keys}"
                )

            # If key was found, add a new prefix with random weights
            self.add_prefix(name=module_prefix_name, cfg=prefix_cfg)

            # Determine apriori how many modules must be loaded from the state dict
            # This is dont to guarentee that partial match does not occur, only exact match
            # between state dict and the prefixs parameters will be allowed.
            modules_to_load = []  # type: List[torch.nn.Module]
            for module in self.modules():
                if (
                    isinstance(module, PrefixModuleMixin)
                    and hasattr(module, 'prefix_name')
                    and module.prefix_name == prefix_name
                ):
                    modules_to_load.append(module)

            # Assert that the number of states in the state dict matches the newly created prefix
            if len(prefix_state) != len(modules_to_load):
                raise ValueError(
                    f"The number of prefixs in current model ({len(modules_to_load)}) does not "
                    f"match the number of modules in the state dict for prefix `{prefix_name}`: "
                    f"({len(prefix_state)})"
                )

            # For the pair of (prefix_state_in_checkpoint, prefix_in_model), restore the weights
            for state, module in zip(prefix_state, modules_to_load):
                module.load_state_dict(state, strict=strict)

            # delete the dictionaries to preserve memory for next prefix
            del prefix_state, modules_to_load

    def update_prefix_cfg(self, cfg: DictConfig):
        """
        Utility method to recursively update all of the Prefix module configs with the provided config.

        .. note::

            It is not a (deep)copy, but a reference copy. Changes made to the config will be reflected to
            prefix submodules, but it is still encouraged to explicitly update the prefix_cfg using this method.

        Args:
            cfg: DictConfig containing the value of `model.cfg.prefixs`.
        """
        for module in self.modules():  # access PT subclass method via inheritance
            if isinstance(module, PrefixModuleMixin):
                module.prefix_cfg = cfg

    @property
    def prefix_module_names(self) -> List[str]:
        """
        List of valid prefix modules that are supported by the model.

        **Note**: Subclasses should override this property and return a list of str names, of all the modules
            that they support, which will enable users to determine where to place the prefix modules.

        Returns:
            A list of str, one for each of the prefix modules that are supported. By default, the subclass
            should support the "global prefix" ('').
        """
        return ['']
