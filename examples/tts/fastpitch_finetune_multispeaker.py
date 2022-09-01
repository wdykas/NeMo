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
import torch
import pytorch_lightning as pl

from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.modules.speaker_modules import Weighted_SpeakerEmbedding
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

from nemo.core import adapter_mixins, lora_mixins, prefix_mixins
from nemo.collections.common.parts import adapter_modules
from omegaconf import DictConfig, open_dict

def update_model_config_to_support_adapter(config) -> DictConfig:
    with open_dict(config):
        enc_adapter_metadata = adapter_mixins.get_registered_adapter(config.input_fft._target_)
        if enc_adapter_metadata is not None:
            config.input_fft._target_ = enc_adapter_metadata.adapter_class_path

        dec_adapter_metadata = adapter_mixins.get_registered_adapter(config.output_fft._target_)
        if dec_adapter_metadata is not None:
            config.output_fft._target_ = dec_adapter_metadata.adapter_class_path
            
        pitch_predictor_adapter_metadata = adapter_mixins.get_registered_adapter(config.pitch_predictor._target_)
        if pitch_predictor_adapter_metadata is not None:
            config.pitch_predictor._target_ = pitch_predictor_adapter_metadata.adapter_class_path
            
        duration_predictor_adapter_metadata = adapter_mixins.get_registered_adapter(config.duration_predictor._target_)
        if duration_predictor_adapter_metadata is not None:
            config.duration_predictor._target_ = duration_predictor_adapter_metadata.adapter_class_path
            
        aligner_adapter_metadata = adapter_mixins.get_registered_adapter(config.alignment_module._target_)
        if aligner_adapter_metadata is not None:
            config.alignment_module._target_ = aligner_adapter_metadata.adapter_class_path
            
    return config

def update_model_config_to_support_lora(config) -> DictConfig:
    with open_dict(config):
        enc_lora_metadata = lora_mixins.get_registered_lora(config.input_fft._target_)
        if enc_lora_metadata is not None:
            config.input_fft._target_ = enc_lora_metadata.lora_class_path

        dec_lora_metadata = lora_mixins.get_registered_lora(config.output_fft._target_)
        if dec_lora_metadata is not None:
            config.output_fft._target_ = dec_lora_metadata.lora_class_path
             
    return config

def update_model_config_to_support_prefix(config) -> DictConfig:
    with open_dict(config):
        enc_prefix_metadata = prefix_mixins.get_registered_prefix(config.input_fft._target_)
        if enc_prefix_metadata is not None:
            config.input_fft._target_ = enc_prefix_metadata.prefix_class_path

        dec_prefix_metadata = prefix_mixins.get_registered_prefix(config.output_fft._target_)
        if dec_prefix_metadata is not None:
            config.output_fft._target_ = dec_prefix_metadata.prefix_class_path
             
    return config


@hydra_runner(config_path="conf", config_name="fastpitch_align_v1.05")
def main(cfg):
    if hasattr(cfg.model.optim, 'sched'):
        logging.warning("You are using an optimizer scheduler while finetuning. Are you sure this is intended?")
    if cfg.model.optim.lr > 1e-3 or cfg.model.optim.lr < 1e-5:
        logging.warning("The recommended learning rate for finetuning is 2e-4")
        
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))   
    n_speakers = cfg.model.n_speakers
    
    if cfg.finetune_multispeaker.add_random_speaker:
        cfg.model.n_speakers += 1
    
    if cfg.finetune_multispeaker.add_adapter:
        model = FastPitchModel(cfg=update_model_config_to_support_adapter(cfg.model), trainer=trainer)
    
    elif cfg.finetune_multispeaker.add_lora:
        model = FastPitchModel(cfg=update_model_config_to_support_lora(cfg.model), trainer=trainer)
    
    elif cfg.finetune_multispeaker.add_prefix:
        model = FastPitchModel(cfg=update_model_config_to_support_prefix(cfg.model), trainer=trainer)
        
    else:
        model = FastPitchModel(cfg=cfg.model, trainer=trainer)
        
    model.fastpitch.speaker_emb = torch.nn.Embedding(n_speakers, model.fastpitch.speaker_emb.embedding_dim)
    model.maybe_init_from_pretrained_checkpoint(cfg=cfg)
    
     # Freeze 
    if cfg.finetune_multispeaker.freeze_all: model.freeze()
    if cfg.finetune_multispeaker.freeze_encoder: model.fastpitch.encoder.freeze()
    if cfg.finetune_multispeaker.freeze_decoder: model.fastpitch.decoder.freeze()
    
    """
    Speaker related modules
    """
    # Add new speaker embedding
    if cfg.finetune_multispeaker.add_random_speaker and model.fastpitch.speaker_emb is not None:
        old_emb = model.fastpitch.speaker_emb
        
        # Choose random
        new_speaker_emb = torch.rand(1, old_emb.embedding_dim)
        # Choose existing
        # new_speaker_emb = old_emb.weight[0, :].unsqueeze(0).detach().clone()
        
        new_emb = torch.nn.Embedding(old_emb.num_embeddings+1, old_emb.embedding_dim).from_pretrained(
            torch.cat([old_emb.weight.detach().clone(), new_speaker_emb], axis=0), freeze=True)
        model.fastpitch.speaker_emb = new_emb
        model.cfg.n_speakers += 1
    
    # Add weighted sum speaker embedding
    elif cfg.finetune_multispeaker.add_weight_speaker and model.fastpitch.speaker_emb is not None:
        old_emb = model.fastpitch.speaker_emb
        new_emb = Weighted_SpeakerEmbedding(pretrained_embedding=old_emb)
        model.fastpitch.speaker_emb = new_emb
    
    # Add conditional layer normalization
    if cfg.finetune_multispeaker.add_cln:
        for name, param in model.named_parameters():
            if 'weight.weight' in name or 'weight.bias' in name \
                or 'bias.weight' in name or 'bias.bias' in name:
                if not param.requires_grad:
                    param.requires_grad = True 
    """
    Parameter-efficient modules
    """
    # Add bitfit
    if cfg.finetune_multispeaker.add_bitfit:
        for name, param in model.named_parameters():
            if 'bias' in name and not param.requires_grad:
                param.requires_grad = True        

    # Add adapter
    if cfg.finetune_multispeaker.add_adapter:
        adapter_cfg = adapter_modules.LinearAdapterConfig(
            in_features=model.cfg.output_fft.d_model,  # conformer specific model dim. Every layer emits this dim at its output.
            dim=256,  # the bottleneck dimension of the adapter
            dropout=0.9,
            activation='swish',  # activation used in bottleneck block
            norm_position='pre',  # whether to use LayerNorm at the beginning or the end of the adapter
        )
        model.add_adapter(name='encoder+decoder+duration_predictor+pitch_predictor+aligner:adapter', cfg=adapter_cfg)
        model.set_enabled_adapters(enabled=False)
        model.set_enabled_adapters('adapter', enabled=True)
        model.unfreeze_enabled_adapters()
    
    # Add lora
    if cfg.finetune_multispeaker.add_lora:
        lora_cfg = adapter_modules.LoraConfig(
            in_features=model.cfg.output_fft.d_model, 
            out_features=model.cfg.output_fft.n_head * model.cfg.output_fft.d_head,
            r=256, 
            alpha=8,
            dropout=0.9,
        )
        model.add_lora(name='encoder+decoder:lora', cfg=lora_cfg)
        model.set_enabled_loras(enabled=False)
        model.set_enabled_loras('lora', enabled=True)
        model.unfreeze_enabled_loras()
        
    # Add prefix
    if cfg.finetune_multispeaker.add_prefix:
        prefix_cfg = adapter_modules.PrefixConfig(
            in_features=model.cfg.output_fft.n_head * model.cfg.output_fft.d_head, 
            dim=256,
            prefix_length=30,
            dropout=0.9,
        )
        model.add_prefix(name='encoder+decoder:prefix', cfg=prefix_cfg)
        model.set_enabled_prefixs(enabled=False)
        model.set_enabled_prefixs('prefix', enabled=True)
        model.unfreeze_enabled_prefixs()
    
    model.summarize()
    
    lr_logger = pl.callbacks.LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([lr_logger, epoch_time_logger])
    trainer.fit(model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
