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

import os
from pathlib import Path

from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.plugins.environments.torchelastic_environment import TorchElasticEnvironment
from pytorch_lightning.plugins.precision.native_amp import NativeMixedPrecisionPlugin
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector

from nemo.collections.nlp.models.language_modeling.megatron_t0_model import MegatronT0SSLPrimeModel
from nemo.collections.nlp.modules.common.megatron.megatron_utils import compute_model_parallel_rank
from nemo.collections.nlp.parts.nlp_overrides import GradScaler, MegatronHalfPrecisionPlugin, NLPDDPPlugin
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import StatelessTimer, exp_manager

#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

@hydra_runner(config_path="conf", config_name="megatron_t0_ssl_prime_config")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    megatron_amp_o2 = cfg.model.get('megatron_amp_O2', False)

    plugins = [
        NLPDDPPlugin(
            no_ddp_communication_hook=(
                    megatron_amp_o2 and cfg.trainer.precision == 'bf16'
            ),  # Only bf16 uses fp32_grad_accum.
            gradient_as_bucket_view=cfg.model.gradient_as_bucket_view,
            find_unused_parameters=False,
        )
    ]
    if cfg.trainer.precision in [16, 'bf16']:
        scaler = None
        if cfg.trainer.precision == 16:
            scaler = GradScaler(
                init_scale=cfg.model.get('native_amp_init_scale', 2 ** 32),
                growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
                hysteresis=cfg.model.get('hysteresis', 2),
            )
        if megatron_amp_o2:
            plugins.append(MegatronHalfPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))
        else:
            plugins.append(NativeMixedPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))

    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    if cfg.model.data.num_data_shards > cfg.trainer.max_epochs:
        logging.info("Datasets are switched every epoch. If num_data_shards > max epoch, "
                     "the model does not train on the whole dataset. "
                     "Switching num_data_shards = max_epoch.")
        cfg.model.data.num_data_shards = cfg.trainer.max_epochs

    trainer = Trainer(
        plugins=plugins,
        reload_dataloaders_every_n_epochs=1 if cfg.model.data.num_data_shards > 1 else 0,
        **cfg.trainer, callbacks=[ModelSummary(max_depth=3)],
    )

    exp_manager(trainer, cfg.exp_manager)

    # update resume from checkpoint found by exp_manager
    resume_from_checkpoint = trainer.checkpoint_connector.resume_from_checkpoint_fit_path
    if resume_from_checkpoint is not None:
        # inject mp_rank into resume_from_checkpoint
        if cfg.model.tensor_model_parallel_size is not None and cfg.model.tensor_model_parallel_size > 1:
            mp_rank = compute_model_parallel_rank(trainer.local_rank, cfg.model.tensor_model_parallel_size)
            resume_from_checkpoint = Path(resume_from_checkpoint)
            resume_from_checkpoint = resume_from_checkpoint.parent.parent.joinpath(f'mp_rank_{mp_rank:02d}').joinpath(
                resume_from_checkpoint.name
            )
            resume_from_checkpoint = str(resume_from_checkpoint)
        logging.info(f'Resuming training from checkpoint: {resume_from_checkpoint}')

    trainer.checkpoint_connector = CheckpointConnector(trainer, resume_from_checkpoint=resume_from_checkpoint)
    # Override timer callback to a stateless one
    for idx, callback in enumerate(trainer.callbacks):
        if isinstance(callback, Timer):
            trainer.callbacks[idx] = StatelessTimer(cfg.trainer.max_time,)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision
    model = MegatronT0SSLPrimeModel(cfg.model, trainer)
    trainer.fit(model)
    trainer.test(model)


if __name__ == '__main__':
    main()
