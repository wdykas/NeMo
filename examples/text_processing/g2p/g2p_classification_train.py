# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import pytorch_lightning as pl
from nemo_text_processing.g2p.models.g2p_classification import G2PClassificationModel

from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager


"""
This script is a copy of g2p_ctc.py with a different config_name value since multi-GPU jobs get stuck when the config_name is updated via command line

To do testing:
python g2p_classification_train.py \
    train_dataset=/home/ebakhturina/g2p_scripts/WikipediaHomographData-master/data/train \
    validation_dataset=/home/ebakhturina/g2p_scripts/WikipediaHomographData-master/data/eval \
    model.encoder.pretrained=/mnt/sdb_4/g2p/chpts/byt5/2982960/g2p/G2PCTC/2022-06-07_01-00-45/checkpoints/G2PCTC.nemo

"""


@hydra_runner(config_path="conf", config_name="g2p_classification.yaml")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    if cfg.do_testing:
        if cfg.pretrained_model and not os.path.exists(cfg.pretrained_model):
            raise ValueError("Pretrained model wasn't found.")
        model = G2PClassificationModel.restore_from(cfg.pretrained_model)
        model.setup_test_data(cfg.model.test_ds)
        trainer.test(model)
    else:
        model = G2PClassificationModel(cfg=cfg.model, trainer=trainer)
        lr_logger = pl.callbacks.LearningRateMonitor()
        epoch_time_logger = LogEpochTimeCallback()
        trainer.callbacks.extend([lr_logger, epoch_time_logger])
        trainer.fit(model)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
