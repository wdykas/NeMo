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

import os
from dataclasses import dataclass
from typing import Optional

import pytorch_lightning as pl
import torch
from nemo_text_processing.g2p.models.g2p_model import G2PModel
from omegaconf import OmegaConf

from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils


"""
python g2p_evaluate.py \
model.test_ds.manifest_filepath=/mnt/sdb/test2/g2p/ipa_v2/phoneme_dev_clean_fields_updated_word_boundaries_ipa.json \
pretrained_model=/mnt/sdb_4/g2p/chpts/conformer/2780434/g2p/G2PCTC/2022-04-04_22-55-15/checkpoints/G2PCTC.nemo

"""


@dataclass
class EvaluationConfig:
    # Required configs
    pretrained_model: str  # Path to a .nemo file or Name of a pretrained model
    manifest_filepath: str  # Path to .json manifest file with "phoneme_field" and "grapheme_field "that match the corresponding fields used during training
    phoneme_field: str = "text"  # name of the field in manifest_filepath for ground truth phonemes
    grapheme_field: str = "text_graphemes"  # name of the field in manifest_filepath for input grapheme text

    # dataloader_params:
    batch_size: int = 20
    num_workers: int = 4


@hydra_runner(config_name="EvaluationConfig", schema=EvaluationConfig)
def main(cfg: EvaluationConfig) -> EvaluationConfig:
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')
    logging.info(
        'During evaluation/testing, it is currently advisable to construct a new Trainer with single GPU and \
            no DDP to obtain accurate results'
    )

    if not cfg.pretrained_model:
        raise ValueError(
            'To run evaluation and inference script a pre-trained model or .nemo file must be provided.'
            f'Choose from {G2PModel.list_available_models()} or "pretrained_model"="your_model.nemo"'
        )

    # setup GPU
    if torch.cuda.is_available():
        device = [0]  # use 0th CUDA device
        accelerator = 'gpu'
    else:
        device = 1
        accelerator = 'cpu'

    map_location = torch.device('cuda:{}'.format(device[0]) if accelerator == 'gpu' else 'cpu')
    trainer = pl.Trainer(devices=device, accelerator=accelerator, logger=False, enable_checkpointing=False)

    if os.path.exists(cfg.pretrained_model):
        # restore model from .nemo file path
        model_cfg = G2PModel.restore_from(restore_path=cfg.pretrained_model, return_config=True)
        classpath = model_cfg.target  # original class path
        imported_class = model_utils.import_class_by_path(classpath)
        logging.info(f"Restoring model : {imported_class.__name__}")
        g2p_model = imported_class.restore_from(restore_path=cfg.pretrained_model, map_location=map_location)
        model_name = os.path.splitext(os.path.basename(cfg.pretrained_model))[0]
        logging.info(f"Restored {model_name} model from {cfg.pretrained_model}.")
    elif cfg.pretrained_model in G2PModel.get_available_model_names():
        # restore model by name
        g2p_model = G2PModel.from_pretrained(cfg.pretrained_model, map_location=map_location)
    else:
        raise ValueError(
            f'Provide path to the pre-trained .nemo checkpoint or choose from {G2PModel.list_available_models()}'
        )

    if not hasattr(g2p_model._cfg, 'test_ds'):
        raise ValueError(f'g2p_model.modeltest_ds was not found in the config, skipping evaluation')

    # update test_ds
    test_ds = g2p_model._cfg.test_ds
    test_ds.manifest_filepath = cfg.manifest_filepath
    test_ds.phoneme_field = cfg.phoneme_field
    test_ds.grapheme_field = cfg.grapheme_field
    test_ds.dataloader_params.batch_size = cfg.batch_size
    test_ds.dataloader_params.num_workers = cfg.num_workers

    # setup trainer
    g2p_model.set_trainer(trainer)
    g2p_model = g2p_model.eval()
    g2p_model.setup_multiple_test_data(test_ds)
    trainer.test(g2p_model)


if __name__ == '__main__':
    main()
