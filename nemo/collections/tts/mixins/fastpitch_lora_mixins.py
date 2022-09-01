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

from nemo.core.classes.mixins.lora_mixins import LoraModelPTMixin, LoraModuleMixin
from nemo.utils import logging, logging_mode


class FastPitchLoraModelMixin(LoraModelPTMixin):
    """ FastPitch Lora Mixin that can augment any Encoder module with Lora module support.
    This mixin class should be used only with a top level ModelPT subclass, that includes an `encoder` submodule.
    This mixin class adds several utility methods which are propagated to the `encoder`.
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
    **Note**: This module **is** responsible for maintaining its config. At the ModelPT level, it will access and
        write Lora config information to `self.cfg.loras`.
    """

    def setup_loras(self):
        """
        Utility method that is called in the ASR ModelPT-implementation constructor, so as to restore any
        loras that were previously added.
        This method should be called just once at constructor time.
        """
        supports_loras = False

        # At least the encoder must extend LoraModuleMixin
        if hasattr(self.fastpitch, 'encoder') and isinstance(self.fastpitch.encoder, LoraModuleMixin):
            supports_loras |= True

        if hasattr(self.fastpitch, 'decoder') and isinstance(self.fastpitch.decoder, LoraModuleMixin):
            supports_loras |= True

      
        # If loras are supported, setup the lora config + any modules (pre-existing lora modules)
        if supports_loras:
            super().setup_loras()

    def add_lora(self, name: str, cfg: DictConfig):
        """
        Add an Lora module to this model.
        Args:
            name: A globally unique name for the lora. Will be used to access, enable and disable loras.
            cfg: A DictConfig that contains at the bare minimum `__target__` to instantiate a new Lora module.
        """
        # setup the config for loras
        super().add_lora(name=name, cfg=cfg)

        # Resolve module name and lora name
        module_name, _ = self.resolve_lora_module_name_(name)

        # Use + as a splitter, in order to share one name across multiple modules
        if '+' in module_name:
            module_names = module_name.split('+')
        else:
            module_names = [module_name]

        # Update the model.cfg with information about the new lora from cfg
        global_config = self._get_global_cfg()
        
        with open_dict(self.cfg):
            for module_name in module_names:
                
                # Check if encoder loras should be added
                if module_name  == 'encoder':
                    # Dispatch the call to the encoder.
                    self.fastpitch.encoder.add_lora(name=name, cfg=cfg)

                # Check if decoder loras should be added
                if module_name in ('', 'decoder'):
                    # Dispatch call to the decoder. (default use decoder)
                    self.fastpitch.decoder.add_lora(name=name, cfg=cfg)
                    

    def is_lora_available(self) -> bool:
        """
        Checks if any Lora module has been instantiated.
        Returns:
            bool, determining if any Lora module has been instantiated. Returns true even if the loras are
            enabled or disabled, false only if no loras exist.
        """
        config_contains_lora = super().is_lora_available()

        # Forward the method call to the individual modules
        if hasattr(self.fastpitch, 'encoder') and isinstance(self.fastpitch.encoder, LoraModuleMixin):
            config_contains_lora |= self.fastpitch.encoder.is_lora_available()

        if hasattr(self.fastpitch, 'decoder') and isinstance(self.fastpitch.decoder, LoraModuleMixin):
            config_contains_lora |= self.fastpitch.decoder.is_lora_available()
                
        return config_contains_lora

    def set_enabled_loras(self, name: Optional[str] = None, enabled: bool = True):
        """
        Updated the internal lora config, determining if an lora (or all loras) are either
        enabled or disabled.
        A common user pattern would be to disable all loras (either after adding them, or restoring a model
        with pre-existing loras) and then simply enable one of the loras.
        .. code::
            model.set_enabled_loras(enabled=False)
            model.set_enabled_loras(name=<some lora name>, enabled=True)
        Args:
            name: Optional str. If a str name is given, the config will be updated to the value of `enabled`.
                If no name is given, then all loras will be enabled/disabled.
            enabled: Bool, determines if the lora(s) will be enabled/disabled.
        """
        super().set_enabled_loras(name=name, enabled=enabled)

        # Resolve the module name and lora name
        if name is not None:
            module_name, _ = self.resolve_lora_module_name_(name)
        else:
            module_name = None

        # Use + as a splitter, in order to share one name across multiple modules
        if module_name is not None and '+' in module_name:
            module_names = module_name.split('+')
        else:
            module_names = [module_name]

        for module_name in module_names:
            # Check if encoder loras should be used
            # Dispatch the call to the encoder.
            if name is None or module_name == 'encoder':
                if self.fastpitch.encoder.is_lora_available():
                    self.fastpitch.encoder.set_enabled_loras(name=name, enabled=enabled)

            # Dispatch the call to the decoder.
            if name is None or module_name in ('', 'decoder'):
                if self.fastpitch.decoder.is_lora_available():
                    self.fastpitch.decoder.set_enabled_loras(name=name, enabled=enabled)
                    
    def get_enabled_loras(self) -> List[str]:
        """
        Returns a list of all enabled loras.
        Returns:
            A list of str names of each enabled lora(s).
        """
        enabled_loras = super().get_enabled_loras()

        # Check if encoder loras should be used or are enabled
        if hasattr(self.fastpitch, 'encoder') and isinstance(self.fastpitch.encoder, LoraModuleMixin):
            enabled_loras.extend(self.fastpitch.encoder.get_enabled_loras())

        if hasattr(self.fastpitch, 'decoder') and isinstance(self.fastpitch.decoder, LoraModuleMixin):
            enabled_loras.extend(self.fastpitch.decoder.get_enabled_loras())

        enabled_loras = list(sorted(list(set(enabled_loras))))

        return enabled_loras

    def check_valid_model_with_lora_support_(self):
        """
        Utility method to test if the subclass of this mixin is an appropriate subclass of ModelPT itself.
        """
        # Obtain the global lora config if possible, otherwise use sensible defaults.
        global_cfg = self._get_global_cfg()

        # Test whether the encoder supports loras
        use_encoder_lora = global_cfg.get('check_encoder_lora', False)
        if use_encoder_lora:
            if not hasattr(self.fastpitch, 'encoder'):
                logging.warning(
                    "Cannot add lora to this object as it does not have an `fastpitch.encoder` sub-module!",
                    mode=logging_mode.ONCE,
                )

            if hasattr(self.fastpitch, 'encoder') and not isinstance(self.fastpitch.encoder, LoraModuleMixin):
                logging.warning(
                    f'{self.fastpitch.encoder.__class__.__name__} does not implement `LoraModuleMixin`',
                    mode=logging_mode.ONCE,
                )

        # Test whether the decoder supports loras
        use_decoder_lora = global_cfg.get('check_decoder_lora', True)
        if use_decoder_lora:
            if not hasattr(self.fastpitch, 'decoder'):
                logging.warning(
                    "Cannot add lora to this object as it does not have an `fastpitch.decoder` sub-module!",
                    mode=logging_mode.ONCE,
                )

            if hasattr(self.fastpitch, 'decoder') and not isinstance(self.fastpitch.decoder, LoraModuleMixin):
                logging.warning(
                    f'{self.fastpitch.decoder.__class__.__name__} does not implement `LoraModuleMixin`',
                    mode=logging_mode.ONCE,
                )
                

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
        module_name, lora_name = super().resolve_lora_module_name_(name)

        # Use + as a splitter, in order to share one name across multiple modules
        if '+' in module_name:
            module_names = module_name.split('+')
        else:
            module_names = [module_name]

        # resolve name and module only for valid modules
        valid_module_names = self.lora_module_names

        for mod_name in module_names:
            if mod_name not in valid_module_names:
                raise ValueError(f"Provided module name `{mod_name}` is not in valid list : {valid_module_names}")

        return (module_name, lora_name)

    def _get_global_cfg(self):
        """
        Utility method, to either extract or construct the global config inside loras config.
        """
        global_config = DictConfig({})
        if 'loras' in self.cfg and self.lora_global_cfg_key in self.cfg.loras:
            global_config = self.lora_cfg[self.lora_global_cfg_key]
        return global_config

    @property
    def lora_module_names(self) -> List[str]:
        module_names = super().lora_module_names  # "Default" lora module: ''
        # Add support for `encoder`, `decoder`, `duration_predictor`, `pitch_predictor` modules
        module_names.extend(['encoder', 'decoder']) 
        return module_names