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

from typing import List, Optional

from omegaconf import DictConfig, open_dict

from nemo.core.classes.mixins.prefix_mixins import PrefixModelPTMixin, PrefixModuleMixin
from nemo.utils import logging, logging_mode


class FastPitchPrefixModelMixin(PrefixModelPTMixin):
    """ FastPitch Prefix Mixin that can augment any Encoder module with Prefix module support.
    This mixin class should be used only with a top level ModelPT subclass, that includes an `encoder` submodule.
    This mixin class adds several utility methods which are propagated to the `encoder`.
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
    **Note**: This module **is** responsible for maintaining its config. At the ModelPT level, it will access and
        write Prefix config information to `self.cfg.prefixs`.
    """

    def setup_prefixs(self):
        """
        Utility method that is called in the ASR ModelPT-implementation constructor, so as to restore any
        prefixs that were previously added.
        This method should be called just once at constructor time.
        """
        supports_prefixs = False

        # At least the encoder must extend PrefixModuleMixin
        if hasattr(self.fastpitch, 'encoder') and isinstance(self.fastpitch.encoder, PrefixModuleMixin):
            supports_prefixs |= True

        if hasattr(self.fastpitch, 'decoder') and isinstance(self.fastpitch.decoder, PrefixModuleMixin):
            supports_prefixs |= True

      
        # If prefixs are supported, setup the prefix config + any modules (pre-existing prefix modules)
        if supports_prefixs:
            super().setup_prefixs()

    def add_prefix(self, name: str, cfg: DictConfig):
        """
        Add an Prefix module to this model.
        Args:
            name: A globally unique name for the prefix. Will be used to access, enable and disable prefixs.
            cfg: A DictConfig that contains at the bare minimum `__target__` to instantiate a new Prefix module.
        """
        # setup the config for prefixs
        super().add_prefix(name=name, cfg=cfg)

        # Resolve module name and prefix name
        module_name, _ = self.resolve_prefix_module_name_(name)

        # Use + as a splitter, in order to share one name across multiple modules
        if '+' in module_name:
            module_names = module_name.split('+')
        else:
            module_names = [module_name]

        # Update the model.cfg with information about the new prefix from cfg
        global_config = self._get_global_cfg()
        
        with open_dict(self.cfg):
            for module_name in module_names:
                
                # Check if encoder prefixs should be added
                if module_name  == 'encoder':
                    # Dispatch the call to the encoder.
                    self.fastpitch.encoder.add_prefix(name=name, cfg=cfg)

                # Check if decoder prefixs should be added
                if module_name in ('', 'decoder'):
                    # Dispatch call to the decoder. (default use decoder)
                    self.fastpitch.decoder.add_prefix(name=name, cfg=cfg)
                    

    def is_prefix_available(self) -> bool:
        """
        Checks if any Prefix module has been instantiated.
        Returns:
            bool, determining if any Prefix module has been instantiated. Returns true even if the prefixs are
            enabled or disabled, false only if no prefixs exist.
        """
        config_contains_prefix = super().is_prefix_available()

        # Forward the method call to the individual modules
        if hasattr(self.fastpitch, 'encoder') and isinstance(self.fastpitch.encoder, PrefixModuleMixin):
            config_contains_prefix |= self.fastpitch.encoder.is_prefix_available()

        if hasattr(self.fastpitch, 'decoder') and isinstance(self.fastpitch.decoder, PrefixModuleMixin):
            config_contains_prefix |= self.fastpitch.decoder.is_prefix_available()
                
        return config_contains_prefix

    def set_enabled_prefixs(self, name: Optional[str] = None, enabled: bool = True):
        """
        Updated the internal prefix config, determining if an prefix (or all prefixs) are either
        enabled or disabled.
        A common user pattern would be to disable all prefixs (either after adding them, or restoring a model
        with pre-existing prefixs) and then simply enable one of the prefixs.
        .. code::
            model.set_enabled_prefixs(enabled=False)
            model.set_enabled_prefixs(name=<some prefix name>, enabled=True)
        Args:
            name: Optional str. If a str name is given, the config will be updated to the value of `enabled`.
                If no name is given, then all prefixs will be enabled/disabled.
            enabled: Bool, determines if the prefix(s) will be enabled/disabled.
        """
        super().set_enabled_prefixs(name=name, enabled=enabled)

        # Resolve the module name and prefix name
        if name is not None:
            module_name, _ = self.resolve_prefix_module_name_(name)
        else:
            module_name = None

        # Use + as a splitter, in order to share one name across multiple modules
        if module_name is not None and '+' in module_name:
            module_names = module_name.split('+')
        else:
            module_names = [module_name]

        for module_name in module_names:
            # Check if encoder prefixs should be used
            # Dispatch the call to the encoder.
            if name is None or module_name == 'encoder':
                if self.fastpitch.encoder.is_prefix_available():
                    self.fastpitch.encoder.set_enabled_prefixs(name=name, enabled=enabled)

            # Dispatch the call to the decoder.
            if name is None or module_name in ('', 'decoder'):
                if self.fastpitch.decoder.is_prefix_available():
                    self.fastpitch.decoder.set_enabled_prefixs(name=name, enabled=enabled)
                    
    def get_enabled_prefixs(self) -> List[str]:
        """
        Returns a list of all enabled prefixs.
        Returns:
            A list of str names of each enabled prefix(s).
        """
        enabled_prefixs = super().get_enabled_prefixs()

        # Check if encoder prefixs should be used or are enabled
        if hasattr(self.fastpitch, 'encoder') and isinstance(self.fastpitch.encoder, PrefixModuleMixin):
            enabled_prefixs.extend(self.fastpitch.encoder.get_enabled_prefixs())

        if hasattr(self.fastpitch, 'decoder') and isinstance(self.fastpitch.decoder, PrefixModuleMixin):
            enabled_prefixs.extend(self.fastpitch.decoder.get_enabled_prefixs())

        enabled_prefixs = list(sorted(list(set(enabled_prefixs))))

        return enabled_prefixs

    def check_valid_model_with_prefix_support_(self):
        """
        Utility method to test if the subclass of this mixin is an appropriate subclass of ModelPT itself.
        """
        # Obtain the global prefix config if possible, otherwise use sensible defaults.
        global_cfg = self._get_global_cfg()

        # Test whether the encoder supports prefixs
        use_encoder_prefix = global_cfg.get('check_encoder_prefix', False)
        if use_encoder_prefix:
            if not hasattr(self.fastpitch, 'encoder'):
                logging.warning(
                    "Cannot add prefix to this object as it does not have an `fastpitch.encoder` sub-module!",
                    mode=logging_mode.ONCE,
                )

            if hasattr(self.fastpitch, 'encoder') and not isinstance(self.fastpitch.encoder, PrefixModuleMixin):
                logging.warning(
                    f'{self.fastpitch.encoder.__class__.__name__} does not implement `PrefixModuleMixin`',
                    mode=logging_mode.ONCE,
                )

        # Test whether the decoder supports prefixs
        use_decoder_prefix = global_cfg.get('check_decoder_prefix', True)
        if use_decoder_prefix:
            if not hasattr(self.fastpitch, 'decoder'):
                logging.warning(
                    "Cannot add prefix to this object as it does not have an `fastpitch.decoder` sub-module!",
                    mode=logging_mode.ONCE,
                )

            if hasattr(self.fastpitch, 'decoder') and not isinstance(self.fastpitch.decoder, PrefixModuleMixin):
                logging.warning(
                    f'{self.fastpitch.decoder.__class__.__name__} does not implement `PrefixModuleMixin`',
                    mode=logging_mode.ONCE,
                )
                

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
        module_name, prefix_name = super().resolve_prefix_module_name_(name)

        # Use + as a splitter, in order to share one name across multiple modules
        if '+' in module_name:
            module_names = module_name.split('+')
        else:
            module_names = [module_name]

        # resolve name and module only for valid modules
        valid_module_names = self.prefix_module_names

        for mod_name in module_names:
            if mod_name not in valid_module_names:
                raise ValueError(f"Provided module name `{mod_name}` is not in valid list : {valid_module_names}")

        return (module_name, prefix_name)

    def _get_global_cfg(self):
        """
        Utility method, to either extract or construct the global config inside prefixs config.
        """
        global_config = DictConfig({})
        if 'prefixs' in self.cfg and self.prefix_global_cfg_key in self.cfg.prefixs:
            global_config = self.prefix_cfg[self.prefix_global_cfg_key]
        return global_config

    @property
    def prefix_module_names(self) -> List[str]:
        module_names = super().prefix_module_names  # "Default" prefix module: ''
        # Add support for `encoder`, `decoder`, `duration_predictor`, `pitch_predictor` modules
        module_names.extend(['encoder', 'decoder']) 
        return module_names