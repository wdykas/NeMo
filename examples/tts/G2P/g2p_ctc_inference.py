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

import json
import os
import sys

from examples.tts.G2P.heteronyms_correction_with_classification import clean, get_metrics, correct_heteronyms

sys.path.append("/home/ebakhturina/NeMo/examples/tts/G2P/data")
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
    pretrained_model=/mnt/sdb_4/g2p/chpts/byt5/2982960/g2p/G2PCTC/2022-06-07_01-00-45/checkpoints/G2PCTC.nemo \
    manifest_filepath=/home/ebakhturina/g2p_scripts/misc_data/phonemes_ipa/dev_clean_ambiguous_checked_fields_updated_word_boundaries_ipa_path.json \
    output_file=dev_clean_ambiguous_checked_fields_updated_word_boundaries_ipa_path_pred.json \
    batch_size=32 \
    num_workers=4 \
    pretrained_heteronyms_model=/home/ebakhturina/NeMo/examples/tts/G2P/nemo_experiments/G2PClassification/2022-06-13_16-49-56/checkpoints/G2PClassification.nemo
    
    target_field=pred_text

python ../../tools/speech_data_explorer/data_explorer.py --disable-caching-metrics dev_clean_ambiguous_checked_fields_updated_word_boundaries_ipa_path_pred.json --port 8052



python G2P/g2p_ctc_inference.py \
pretrained_model=/mnt/sdb_4/g2p/chpts/byt5/3043007_cased_with_punct/g2p/G2PCTC/2022-06-22_00-20-21/checkpoints/G2PCTC.nemo \
manifest_filepath=/mnt/sdb_4/g2p/data_ipa/with_unicode_token/lower_False_Truepunct/eval_wikihomograph.json \
output_file=cases_with_punct_eval_wiki.json pretrained_heteronyms_model=/mnt/sdb_4/g2p/chpts/homographs_classification/G2PClassification.nemo 


"""


@dataclass
class TranscriptionConfig:
    # Required configs
    pretrained_model: str  # Path to a .nemo file or Name of a pretrained model
    manifest_filepath: str  # Path to .json manifest file with "text" field
    pretrained_heteronyms_model: Optional[
        str
    ] = None  # Path to a .nemo file or a Name of a pretrained model to disambiguage heteronyms (Optional)
    clean_word_level: bool = False
    clean_sent_level: bool = False

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
    model._cfg.max_source_len = 512
    model.set_trainer(trainer)
    model = model.eval()

    if cfg.output_file is None:
        cfg.output_file = cfg.manifest_filepath.replace(".json", "_phonemes.json")

    if cfg.clean_sent_level or cfg.clean_word_level:
        with open(cfg.manifest_filepath, "r") as f_in, open(cfg.output_file, "w") as f_out:
            for line in f_in:
                line = json.loads(line)

                clean_graphemes = clean(line["text_graphemes"])
                clean_phonemes = clean(line["text"])
                line["original_text"] = line["text"]
                line["text"] = clean_phonemes

                tmp_file = "/tmp/tmp.json"
                with open(tmp_file, "w") as f_tmp:
                    if cfg.clean_word_level:
                        clean_graphemes = clean_graphemes.split()
                        for g in clean_graphemes:
                            entry = {"text_graphemes": g}
                            f_tmp.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    else:
                        # for sentence level:
                        entry = {"text_graphemes": clean_graphemes}
                        f_tmp.write(json.dumps(entry, ensure_ascii=False) + "\n")

                with torch.no_grad():
                    preds = model.convert_graphemes_to_phonemes(
                        manifest_filepath=tmp_file,
                        output_manifest_filepath=tmp_file.replace(".json", "_preds.json"),
                        batch_size=cfg.batch_size,
                        num_workers=cfg.num_workers,
                        target_field=cfg.target_field,
                        verbose = False
                    )

                preds = " ".join(preds)
                line["pred_text"] = preds
                line["clean_graphemes"] = clean_graphemes
                f_out.write(json.dumps(line, ensure_ascii=False) + "\n")

        get_metrics(cfg.output_file)
        exit()

    with torch.no_grad():
        model.convert_graphemes_to_phonemes(
            manifest_filepath=cfg.manifest_filepath,
            output_manifest_filepath=cfg.output_file,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            target_field=cfg.target_field,
        )
        print(f"IPA predictions saved in {cfg.output_file}")
        get_metrics(cfg.output_file)

    if cfg.pretrained_heteronyms_model is not None:
        correct_heteronyms(cfg.pretrained_heteronyms_model, cfg.output_file, cfg.batch_size, cfg.num_workers)




if __name__ == '__main__':
    main()


"""
python G2P/g2p_ctc_inference.py \
pretrained_model=/mnt/sdb_4/g2p/chpts/byt5/3043007_cased_with_punct/g2p/G2PCTC/2022-06-22_00-20-21/checkpoints/G2PCTC.nemo manifest_filepath=sample.json output_file=sample_PRED.json pretrained_heteronyms_model=/mnt/sdb_4/g2p/chpts/homographs_classification/G2PClassification.nemo 


/mnt/sdb_4/g2p/data_ipa/evaluation_sets/dev_cmu_phonemes.json: WER: 82.69%, PER: 35.66%

"""
