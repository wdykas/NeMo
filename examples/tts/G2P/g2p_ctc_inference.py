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
from dataclasses import dataclass, is_dataclass
from typing import Optional

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from nemo.collections.tts.models.G2P.g2p_ctc import CTCG2PModel
from nemo.core.config import hydra_runner
from nemo.utils import logging


"""
python G2P/g2p_ctc_inference.py \
    pretrained_model=/mnt/sdb_4/g2p/chpts/conformer/2780434/g2p/G2PCTC/2022-04-04_22-55-15/checkpoints/G2PCTC.nemo \
    manifest_filepath=manifest_filepath=/home/ebakhturina/g2p_scripts/misc_data/phonemes_ipa/dev_clean_ambiguous_checked_fields_updated_word_boundaries_ipa_path.json \
    batch_size=32 \
    num_workers=4 \
    target_field=pred_text

python ../../tools/speech_data_explorer/data_explorer.py /mnt/sdb_4/g2p/v2/phoneme_dev_clean_phonemes.json --disable-caching-metrics --port 8052

"""


@dataclass
class TranscriptionConfig:
    # Required configs
    pretrained_model: str  # Path to a .nemo file or Name of a pretrained model
    manifest_filepath: str  # Path to .json manifest file with "text" field

    # General configs
    output_file: Optional[str] = None
    target_field: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 0


@hydra_runner(config_name="TranscriptionConfig", schema=TranscriptionConfig)
def main(cfg: TranscriptionConfig) -> TranscriptionConfig:
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if not cfg.pretrained_model:
        raise ValueError(
            'To run evaluation and inference script a pre-trained model or .nemo file must be provided.'
            f'Choose from {CTCG2PModel.list_available_models()} or "pretrained_model"="your_model.nemo"'
        )

    logging.info(
        'During evaluation/testing, it is currently advisable to construct a new Trainer with single GPU and \
            no DDP to obtain accurate results'
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
        model = CTCG2PModel.restore_from(cfg.pretrained_model, map_location=map_location)
    elif cfg.pretrained_model in CTCG2PModel.get_available_model_names():
        model = CTCG2PModel.from_pretrained(cfg.pretrained_model, map_location=map_location)
    else:
        raise ValueError(
            f'Provide path to the pre-trained .nemo checkpoint or choose from {CTCG2PModel.list_available_models()}'
        )
    model.set_trainer(trainer)
    model = model.eval()

    #
    # if cfg.do_testing:
    #     import pdb; pdb.set_trace()
    #     if not hasattr(cfg.model, 'test_ds'):
    #         logging.error(f'model.test_ds was not found in the config, skipping evaluation')
    #     elif model.prepare_test(trainer):
    #         model.setup_test_data(cfg.model.test_ds)
    #         trainer.test(model)
    # else:
    #     logging.error('Skipping the evaluation. The trainer is not setup properly.')

    with torch.no_grad():
        model.convert_graphemes_to_phonemes(
            manifest_filepath=cfg.manifest_filepath,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            output_file=cfg.output_file,
            target_field=cfg.target_field,
        )


if __name__ == '__main__':
    main()
