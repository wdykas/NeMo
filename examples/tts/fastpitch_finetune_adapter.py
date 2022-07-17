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
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

from nemo.core import adapter_mixins
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
            
    return config

@hydra_runner(config_path="conf", config_name="fastpitch_align_44100")
def main(cfg):
    if hasattr(cfg.model.optim, 'sched'):
        logging.warning("You are using an optimizer scheduler while finetuning. Are you sure this is intended?")
    if cfg.model.optim.lr > 1e-3 or cfg.model.optim.lr < 1e-5:
        logging.warning("The recommended learning rate for finetuning is 2e-4")
        
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))    
    cfg.model.n_speakers += 1
    model = FastPitchModel(cfg=update_model_config_to_support_adapter(cfg.model), trainer=trainer)
    model.fastpitch.speaker_emb = torch.nn.Embedding(model.fastpitch.speaker_emb.num_embeddings-1, model.fastpitch.speaker_emb.embedding_dim)
    model.maybe_init_from_pretrained_checkpoint(cfg=cfg)
    
     # Freeze 
    if cfg.finetune.freeze_encoder: model.fastpitch.encoder.freeze()
    if cfg.finetune.freeze_decoder: model.fastpitch.decoder.freeze()
    if cfg.finetune.freeze_all: model.freeze()
    
    # Add new speaker embedding
    if cfg.finetune.add_speaker_embedding and model.fastpitch.speaker_emb is not None:
        old_emb = model.fastpitch.speaker_emb
        # new_speaker_emb = torch.rand(1, old_emb.embedding_dim)
        new_speaker_emb = old_emb.weight[3, :].unsqueeze(0).detach().clone()
        
        new_emb = torch.nn.Embedding(old_emb.num_embeddings+1, old_emb.embedding_dim).from_pretrained(
            torch.cat([old_emb.weight.detach().clone(), new_speaker_emb], axis=0), freeze=False)
        model.fastpitch.speaker_emb = new_emb
        model.cfg.n_speakers += 1
        
    # Add adapter
    if cfg.finetune.add_adapter:
        adapter_cfg = adapter_modules.LinearAdapterConfig(
            in_features=model.cfg.output_fft.d_model,  # conformer specific model dim. Every layer emits this dim at its output.
            dim=256,  # the bottleneck dimension of the adapter
            activation='swish',  # activation used in bottleneck block
            norm_position='post',  # whether to use LayerNorm at the beginning or the end of the adapter
        )
        model.add_adapter(name='encoder+decoder+duration_predictor+pitch_predictor:adapter', cfg=adapter_cfg)
        # model.add_adapter(name='encoder+decoder:adapter', cfg=adapter_cfg)
        model.set_enabled_adapters(enabled=False)
        model.set_enabled_adapters('adapter', enabled=True)
        model.unfreeze_enabled_adapters()
   
    model.summarize()
    
    lr_logger = pl.callbacks.LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    trainer.callbacks.extend([lr_logger, epoch_time_logger])
    trainer.fit(model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
