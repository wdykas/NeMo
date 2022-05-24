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

"""
This script is a copy of g2p_ctc.py with a different config_name value since multi-GPU jobs get stuck when the config_name is updated via command line

To do testing:
python g2p_ctc.py \
    do_testing=True \
    model.test_ds.dataset.manifest_filepath=/mnt/sdb_4/g2p/v2/phoneme_dev_clean.json \
    pretrained_model=/mnt/sdb_4/g2p/chpts/2552576/g2p/G2PCTC/2022-01-28_01-00-25/checkpoints
"""

import os

import pytorch_lightning as pl

from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.models import CTCG2PModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager


@hydra_runner(config_path="../conf/G2P", config_name="g2p_t5_ctc.yaml")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    if cfg.do_testing:
        if cfg.pretrained_model and not os.path.exists(cfg.pretrained_model):
            raise ValueError("Pretrained model wasn't found.")
        model = CTCG2PModel.restore_from(cfg.pretrained_model)
        model.setup_test_data(cfg.model.test_ds)
        trainer.test(model)
    else:
        model = CTCG2PModel(cfg=cfg.model, trainer=trainer)
        lr_logger = pl.callbacks.LearningRateMonitor()
        epoch_time_logger = LogEpochTimeCallback()
        trainer.callbacks.extend([lr_logger, epoch_time_logger])
        trainer.fit(model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
