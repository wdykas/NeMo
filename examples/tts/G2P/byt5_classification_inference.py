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


import csv
import os
from dataclasses import dataclass, is_dataclass
from glob import glob
from typing import Optional

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.collections.tts.models import G2PClassificationModel
from nemo.collections.tts.models.G2P.g2p_classification import G2PClassificationModel
from nemo.collections.tts.torch.g2p_classification_data import read_wikihomograph_file
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager


"""
python byt5_classification_inference.py \
pretrained_model=/mnt/sdb_4/g2p/chpts/homographs_classification/G2PClassification.nemo \
data_dir=/home/ebakhturina/g2p_scripts/WikipediaHomographData-master/data/eval

filepath=/home/ebakhturina/g2p_scripts/WikipediaHomographData-master/data/eval/read.tsv

Accuracy: 98.89% (18 errors out of 1615)
"""


@dataclass
class TranscriptionConfig:
    # Required configs
    pretrained_model: str  # Path to a .nemo file or Name of a pretrained model

    # specify either filepath or data_dir
    filepath: Optional[str] = None  # Path to .tsv file
    data_dir: Optional[str] = None  # Path to a directory with .tsv file

    # General configs
    output_file: Optional[str] = None
    batch_size: int = 32
    num_workers: int = 0


@hydra_runner(config_name="TranscriptionConfig", schema=TranscriptionConfig)
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    if is_dataclass(cfg):
        cfg = OmegaConf.structured(cfg)

    if not cfg.pretrained_model:
        raise ValueError(
            'To run evaluation and inference script a pre-trained model or .nemo file must be provided.'
            f'Choose from {G2PClassificationModel.list_available_models()} or "pretrained_model"="your_model.nemo"'
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
        model = G2PClassificationModel.restore_from(cfg.pretrained_model, map_location=map_location)
    elif cfg.pretrained_model in G2PClassificationModel.get_available_model_names():
        model = G2PClassificationModel.from_pretrained(cfg.pretrained_model, map_location=map_location)
    else:
        raise ValueError(
            f'Provide path to the pre-trained .nemo checkpoint or choose from {G2PClassificationModel.list_available_models()}'
        )
    model.set_trainer(trainer)
    model = model.eval()

    sentences = []
    start_end_indices = []
    homographs = []
    word_ids = []

    if cfg.data_dir and os.path.exists(cfg.data_dir):
        for file in sorted(glob(f"{cfg.data_dir}/*.tsv")):
            sentences_, start_end_indices_, homographs_, word_ids_ = read_wikihomograph_file(file)
            sentences.extend(sentences_)
            start_end_indices.extend(start_end_indices_)
            homographs.extend(homographs_)
            word_ids.extend(word_ids_)
    elif cfg.filepath and os.path.exists(cfg.filepath):
        sentences, start_end_indices, homographs, word_ids = read_wikihomograph_file(cfg.filepath)
    else:
        raise ValueError("Data dir or filepath not found.")

    with torch.no_grad():
        preds = model.disambiguate(
            sentences=sentences,
            start_end_indices=start_end_indices,
            homographs=homographs,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )

    target_ipa_id_to_label = {v: k for k, v in model.target_ipa_label_to_id.items()}
    correct = 0
    errors_f = "errors.txt"
    with open(errors_f, "w", encoding="utf-8") as f_out:
        for i, pred in enumerate(preds):
            if target_ipa_id_to_label[pred] != word_ids[i]:
                f_out.write(f"INPUT: {sentences[i]}\n")
                f_out.write(f"PRED : {target_ipa_id_to_label[pred]} not {word_ids[i]}\n")
                f_out.write("===========================\n")
            else:
                correct += 1

    print(f"Accuracy: {round(correct/len(preds) * 100, 2)}% ({len(preds) - correct} errors out of {len(preds)})")
    print(f"Errors saved at {errors_f}")


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
