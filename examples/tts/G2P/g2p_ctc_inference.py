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
from dataclasses import dataclass, is_dataclass
from typing import Optional

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from nemo.collections.tts.models.G2P.g2p_classification import G2PClassificationModel
from nemo.collections.tts.models.G2P.g2p_ctc import CTCG2PModel
from nemo.collections.tts.torch.g2p_classification_data import read_wordids
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

python ../../tools/speech_data_explorer/data_explorer.py dev_clean_ambiguous_checked_fields_updated_word_boundaries_ipa_path_pred.json --disable-caching-metrics --port 8052

WER 1.64%
CER 0.57%


/mnt/sdb_4/g2p/chpts/conformer/BPE/3018285
WER 1.69%
CER 0.61%

1.64
0.57
"""


@dataclass
class TranscriptionConfig:
    # Required configs
    pretrained_model: str  # Path to a .nemo file or Name of a pretrained model
    manifest_filepath: str  # Path to .json manifest file with "text" field
    pretrained_heteronyms_model: Optional[
        str
    ] = None  # Path to a .nemo file or a Name of a pretrained model to disambiguage heteronyms (Optional)

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

    if cfg.pretrained_heteronyms_model is not None:
        if os.path.exists(cfg.pretrained_heteronyms_model):
            heteronyms_model = G2PClassificationModel.restore_from(
                cfg.pretrained_heteronyms_model, map_location=map_location
            )
        elif cfg.pretrained_model in G2PClassificationModel.get_available_model_names():
            heteronyms_model = G2PClassificationModel.from_pretrained(
                cfg.pretrained_heteronyms_model, map_location=map_location
            )
        else:
            raise ValueError(
                f'Invalid path to the pre-trained .nemo checkpoint or model name for G2PClassificationModel'
            )
        heteronyms_model.set_trainer(trainer)
        heteronyms_model = heteronyms_model.eval()

        # load heteronyms list
        heteronyms_path = "/home/ebakhturina/NeMo/scripts/tts_dataset_files/heteronyms-052722"
        heteronyms = []
        with open(heteronyms_path, encoding="utf-8", mode="r") as f:
            for line in f:
                heteronyms.append(line.strip())

        target_ipa_id_to_label = {v: k for k, v in heteronyms_model.target_ipa_label_to_id.items()}
        # TODO: remove hardcoding, save to .nemo
        wiki_homograph_dict, target_ipa, target_ipa_label_to_id = read_wordids(
            "/home/ebakhturina/g2p_scripts/WikipediaHomographData-master/data/wordids.tsv"
        )
        grapheme_unk_token = "<unk>"
        ipa_unk_token = "§"
        (
            manifest_filepath_with_unk,
            sentences,
            sent_ids,
            start_end_indices,
            homographs,
            heteronyms_ipa,
        ) = add_unk_token_to_manifest(
            cfg.manifest_filepath, heteronyms, wiki_homograph_dict, grapheme_unk_token="<unk>", ipa_unk_token="§"
        )

    # if cfg.do_testing:
    #     import pdb; pdb.set_trace()
    #     if not hasattr(cfg.model, 'test_ds'):
    #         logging.error(f'model.test_ds was not found in the config, skipping evaluation')
    #     elif model.prepare_test(trainer):
    #         model.setup_test_data(cfg.model.test_ds)
    #         trainer.test(model)
    # else:
    #     logging.error('Skipping the evaluation. The trainer is not setup properly.')

    if cfg.output_file is None:
        cfg.output_file = cfg.manifest_filepath.replace(".json", "_phonemes.json")

    with torch.no_grad():
        model.convert_graphemes_to_phonemes(
            manifest_filepath=cfg.manifest_filepath,
            output_manifest_filepath=cfg.output_file,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            target_field=cfg.target_field,
        )
        print(f"IPA predictions saved in {cfg.output_file}")

        output_file_with_unk = cfg.output_file.replace(".json", "_with_unk.json")
        model.convert_graphemes_to_phonemes(
            manifest_filepath=manifest_filepath_with_unk,
            output_manifest_filepath=output_file_with_unk,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            target_field=cfg.target_field,
        )

    if cfg.pretrained_heteronyms_model:
        with torch.no_grad():
            heteronyms_preds = heteronyms_model.disambiguate(
                sentences=sentences,
                start_end_indices=start_end_indices,
                homographs=homographs,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
            )

        wordid_to_nemo_cmu = {}
        with open("/home/ebakhturina/g2p_scripts/misc_data/wordid_to_nemo_cmu.tsv", "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                line = line.strip().split("\t")
                wordid_to_nemo_cmu[line[0]] = line[1]

        # replace <unk> token
        heteronyms_sent_id = 0
        # TODO fall back if the UNK token is not predicted -> could lead to mismatch
        manifest_corrected_heteronyms = cfg.output_file.replace(".json", "_corrected_heteronyms.json")
        with open(output_file_with_unk, "r", encoding="utf-8") as f_in, open(
            manifest_corrected_heteronyms, "w", encoding="utf-8"
        ) as f_out:
            for idx, line in enumerate(f_in):
                line = json.loads(line)
                line["text"] = line["text_original"]

                if idx in sent_ids:
                    count_unk = line["text_graphemes"].count(grapheme_unk_token)
                    for i in range(count_unk):
                        word_id = target_ipa_id_to_label[heteronyms_preds[heteronyms_sent_id]]
                        ipa_pred = wordid_to_nemo_cmu[word_id]

                        # replace first occurance of the unk_token
                        idx_unk_token = line["pred_text"].find(ipa_unk_token)
                        line["pred_text"] = (
                            line["pred_text"][: idx_unk_token + 1].replace(ipa_unk_token, ipa_pred)
                            + line["pred_text"][idx_unk_token + 1 :]
                        )
                        heteronyms_sent_id += 1
                f_out.write(json.dumps(line, ensure_ascii=False) + "\n")

        print(f"{heteronyms_sent_id} replacements done out of {len(sentences)}")
        print(f"Saved in {manifest_corrected_heteronyms}")


def add_unk_token_to_manifest(
    manifest, heteronyms, wiki_homograph_dict, grapheme_unk_token="<unk>", ipa_unk_token="§"
):
    """ TODO: update to handle text with punctuation """
    tmp_manifest = f"/tmp/{os.path.basename(manifest)}"

    sentences = []
    start_end_indices = []
    homographs = []
    heteronyms_ipa = []
    sent_ids = []

    with open(manifest, "r", encoding="utf-8") as f_in, open(tmp_manifest, "w", encoding="utf-8",) as f_out:
        for sent_id, line in enumerate(f_in):
            line = json.loads(line)
            ipa = line["text"].split()
            graphemes = line["text_graphemes"].split()
            line["text_graphemes_original"] = line["text_graphemes"]
            line["text_original"] = line["text"]

            assert len(graphemes) == len(ipa)

            graphemes_with_unk = []
            ipa_with_unk = []

            heteronym_present = False
            for i, word in enumerate(graphemes):
                if word.lower() in heteronyms and word.lower() in wiki_homograph_dict:
                    graphemes_with_unk.append(grapheme_unk_token)
                    ipa_with_unk.append(ipa_unk_token)

                    # collect data for G2P classification
                    start_idx = line["text_graphemes"].index(word, len(" ".join(graphemes[:i])))
                    end_idx = start_idx + len(word)
                    start_end_indices.append([start_idx, end_idx])
                    sentences.append(line["text_graphemes"])
                    homographs.append(word.lower())
                    heteronyms_ipa.append(ipa[i])
                    heteronym_present = True
                    sent_ids.append(sent_id)
                else:
                    graphemes_with_unk.append(word)
                    ipa_with_unk.append(ipa[i])

            if heteronym_present:
                line["text_graphemes"] = " ".join(graphemes_with_unk)
                line["text"] = " ".join(ipa_with_unk)

            f_out.write(json.dumps(line, ensure_ascii=False) + '\n')
    return tmp_manifest, sentences, sent_ids, start_end_indices, homographs, heteronyms_ipa


if __name__ == '__main__':
    main()
