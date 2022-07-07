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
from dataclasses import dataclass, is_dataclass
from typing import Optional

import pytorch_lightning as pl
import torch
from data_preparation_utils import remove_punctuation
from omegaconf import OmegaConf
from tqdm import tqdm

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.tts.models.G2P.g2p_classification import G2PClassificationModel
from nemo.collections.tts.models.G2P.g2p_ctc import CTCG2PModel
from nemo.collections.tts.torch.en_utils import english_word_tokenize
from nemo.collections.tts.torch.g2p_classification_data import read_wordids
from nemo.core.config import hydra_runner
from nemo.utils import logging

sys.path.append("/home/ebakhturina/NeMo/examples/tts/G2P/data")


# fmt: off
sys.path.append("/home/ebakhturina/NeMo/examples/tts/G2P/data")

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

    # disambiguate heteronyms using IPA-classification model
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
        grapheme_unk_token = "҂"
        ipa_unk_token = "҂"
        (
            sentences,
            sentence_id_to_meta_info,
            start_end_indices_graphemes,
            homographs,
            heteronyms_ipa,
        ) = add_unk_token_to_manifest(
            cfg.output_file,
            heteronyms,
            wiki_homograph_dict,
            grapheme_unk_token=grapheme_unk_token,
            ipa_unk_token=ipa_unk_token,
        )

        with torch.no_grad():
            heteronyms_preds = heteronyms_model.disambiguate(
                sentences=sentences,
                start_end_indices=start_end_indices_graphemes,
                homographs=homographs,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
            )

        wordid_to_nemo_cmu = get_wordid_to_nemo()

        # replace unknown token
        heteronyms_sent_id = 0
        # TODO fall back if the UNK token is not predicted -> could lead to mismatch
        manifest_corrected_heteronyms = cfg.output_file.replace(".json", "_corrected_heteronyms.json")

        with open(cfg.output_file, "r", encoding="utf-8") as f_default, open(
            manifest_corrected_heteronyms, "w", encoding="utf-8"
        ) as f_corrected:
            for sent_id, line in enumerate(f_default):
                line = json.loads(line)

                corrected_pred_text = line["pred_text"]
                offset = 0
                if sent_id in sentence_id_to_meta_info:
                    for start_end_idx in sentence_id_to_meta_info[sent_id]["unk_indices"]:
                        start_idx, end_idx = start_end_idx
                        word_id = target_ipa_id_to_label[heteronyms_preds[heteronyms_sent_id]]
                        ipa_pred = wordid_to_nemo_cmu[word_id]
                        if sent_id == 0 and start_idx == 0:
                            ipa_pred = ipa_pred[:-2]
                        heteronyms_sent_id += 1
                        start_idx += offset
                        end_idx += offset
                        corrected_pred_text = (
                            corrected_pred_text[:start_idx] + ipa_pred + corrected_pred_text[end_idx:]
                        )
                        # offset to correct indices, since ipa form chosen during classification could differ in length
                        # from the predicted one
                        offset += len(ipa_pred) - (end_idx - start_idx)

                line["pred_text"] = corrected_pred_text
                f_corrected.write(json.dumps(line, ensure_ascii=False) + "\n")

        print(f"Saved in {manifest_corrected_heteronyms}")
        get_metrics(manifest_corrected_heteronyms)
        cfg.output_file = manifest_corrected_heteronyms

    # if cfg.clean:
    #     clean_file = cfg.output_file.replace(".json", "_clean.json")
    #     with open(cfg.output_file, "r") as f_in, open(clean_file, "w") as f_out:
    #         for line in tqdm(f_in):
    #             line = json.loads(line)
    #
    #             clean_references = clean(line["text"])
    #             clean_phonemes_preds = clean(line["pred_text"])
    #             line["text_original"] = line["text"]
    #             line["text"] = clean_references
    #             line["pred_text"] = clean_phonemes_preds
    #             f_out.write(json.dumps(line, ensure_ascii=False) + "\n")
    #
    #     get_metrics(clean_file)

def get_wordid_to_nemo():
    wordid_to_nemo_cmu = {}
    with open("/home/ebakhturina/g2p_scripts/misc_data/wordid_to_nemo_cmu.tsv", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            line = line.strip().split("\t")
            wordid_to_nemo_cmu[line[0]] = line[1]
    return wordid_to_nemo_cmu

def _get_ipa_parts(words_per_segment, cur_ipa):
    try:
        current_ipa = []
        ipa_end_idx = 0
        add_previous = False
        for k, word in enumerate(words_per_segment):
            if word.isalpha():
                if add_previous:
                    import pdb

                    pdb.set_trace()
                    raise ValueError()
                if k == len(words_per_segment) - 1:
                    current_ipa.append(cur_ipa[ipa_end_idx:])
                else:
                    add_previous = True
            else:
                if add_previous:
                    new_ipa_end_idx = cur_ipa.index(word, ipa_end_idx)
                    current_ipa.append(cur_ipa[ipa_end_idx:new_ipa_end_idx])
                    ipa_end_idx = new_ipa_end_idx
                    add_previous = False

                word_len = len(word)
                current_ipa.append(cur_ipa[ipa_end_idx : ipa_end_idx + word_len])
                ipa_end_idx += word_len

        assert len(current_ipa) == len(words_per_segment)
    except Exception as e:
        return None
    return current_ipa


def add_unk_token_to_manifest(manifest, heteronyms, wiki_homograph_dict, grapheme_unk_token, ipa_unk_token):
    """ Replace heteronyms in graphemes ("text_grapheme") and their ipa prediction () with UNKNOWN token"""
    sentences = []
    # here we're going to store sentence id (in the originla manifest file) and homograph start indices present
    # in this sentences. These values willbe later use to replace unknown tokens with classification model predictions.
    sentence_id_to_meta_info = {}
    start_end_indices_graphemes = []
    homographs = []
    heteronyms_ipa = []

    with open(manifest, "r", encoding="utf-8") as f_in:
        for sent_id, line in enumerate(f_in):
            line = json.loads(line)
            ipa = line["pred_text"].split()
            graphemes = line["text_graphemes"]
            line["text_graphemes_original"] = line["text_graphemes"]
            line["pred_text_default"] = line["pred_text"]

            if len(graphemes.split()) != len(ipa):
                print(f"len(graphemes+ != len(ipa), {line}, skipping...")
                continue

            graphemes_with_unk = ""
            ipa_with_unk = ""
            heteronym_present = False
            for i, segment in enumerate(graphemes.split()):
                cur_graphemes_with_unk = []
                cur_ipa_with_unk = []
                words_per_segment = [w for w, _ in english_word_tokenize(segment, lower=False)]

                # get ipa segments that match words, i.e. split punctuation marks from words
                current_ipa = _get_ipa_parts(words_per_segment, ipa[i])

                # unmatched ipa and graphemes, e.g. incorrect punctuation mark predicted
                if current_ipa is None:
                    cur_graphemes_with_unk.append(segment)
                    cur_ipa_with_unk.append(ipa[i])
                else:
                    for j, word in enumerate(words_per_segment):
                        if word.lower() in heteronyms and word.lower() in wiki_homograph_dict:
                            cur_graphemes_with_unk.append(grapheme_unk_token)
                            cur_ipa_with_unk.append(ipa_unk_token)

                            # collect data for G2P classification
                            # start and end indices for heteronyms in grapheme form
                            start_idx_grapheme = line["text_graphemes"].index(
                                word, len(" ".join(graphemes.split()[:i]))
                            )
                            end_idx_grapheme = start_idx_grapheme + len(word)
                            start_end_indices_graphemes.append([start_idx_grapheme, end_idx_grapheme])

                            # start and end indices for heteronyms in ipa form ("pred_text" field to later replace
                            # unk token with classification model prediction)
                            start_idx_ipa = line["pred_text"].index(current_ipa[j], len(" ".join(ipa[:i])))
                            end_idx_ipa = start_idx_ipa + len(current_ipa[j])

                            if sent_id not in sentence_id_to_meta_info:
                                sentence_id_to_meta_info[sent_id] = {}
                                sentence_id_to_meta_info[sent_id]["unk_indices"] = []
                                sentence_id_to_meta_info[sent_id]["default_pred"] = []

                            sentence_id_to_meta_info[sent_id]["unk_indices"].append([start_idx_ipa, end_idx_ipa])
                            sentence_id_to_meta_info[sent_id]["default_pred"].append(current_ipa[j])

                            sentences.append(graphemes)
                            homographs.append(word.lower())
                            heteronym_present = True
                        else:
                            cur_graphemes_with_unk.append(word)
                            cur_ipa_with_unk.append(current_ipa[j])

                graphemes_with_unk += " " + "".join(cur_graphemes_with_unk)

                ipa_with_unk += " " + "".join(cur_ipa_with_unk)
                cur_ipa_with_unk = []
                cur_grapheme_unk_token = []

            if heteronym_present:
                line["text_graphemes"] = graphemes_with_unk.strip()
    return (
        sentences,
        sentence_id_to_meta_info,
        start_end_indices_graphemes,
        homographs,
        heteronyms_ipa,
    )


def clean(text, do_lower=True):
    exclude_punct = "'ˈˌ-'"
    if do_lower:
        text = text.lower()
    # text = (
    #     text.replace("long-term", "longterm")
    #     .replace(" x-ray ", " xray ")
    #     .replace("t-shirts", "tshirts")
    #     .replace("twenty-five", "twentyfive")
    #     .replace("spider-man", "spiderman")
    #     .replace("full-time", "fulltime")
    #     .replace("three-year", "threeyear")
    #     .replace("one-year", "oneyear")
    #     .replace("five-year", "fiveyear")
    #     .replace("two-dimensional", "twodimensional")
    #     .replace("three-dimensional", "threedimensional")
    #     .replace("computer-generated", "computergenerated")
    #     .replace("long-range", "longrange")
    #     .replace("non-zero", "nonzero")
    #     .replace("air-conditioning", "airconditioning")
    #     .replace("pre-season", "preseason")
    #     .replace("build-up", "buildup")
    #     .replace("one-off", "oneoff")
    #     .replace("brother-in-law", "brotherinlaw")
    #     .replace("on-screen", "onscreen")
    #     .replace("non-verbal", "nonverbal")
    #     .replace("all-out", "allout")
    # )
    text = remove_punctuation(text, exclude=exclude_punct)
    text = text.replace("҂", "").replace("  ", " ").replace("  ", " ").strip()
    return text

def get_metrics(manifest: str):
    all_preds = []
    all_references = []
    all_graphemes = {}
    with open(manifest, "r") as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            all_preds.append(line["pred_text"])
            all_references.append(line["text"])

            if line["text_graphemes"] not in all_graphemes:
                all_graphemes[line["text_graphemes"]] = []
            all_graphemes[line["text_graphemes"]].append(i)

    # collect all examples with multiple phoneme options and same grapheme form, choose the one with min PER
    all_graphemes = {k: v for k, v in all_graphemes.items() if len(v) > 1}
    lines_to_drop = []
    for phon_amb_indices  in all_graphemes.values():
        refs = all_references[phon_amb_indices[0]:phon_amb_indices[-1] + 1]
        preds = all_preds[phon_amb_indices[0]:phon_amb_indices[-1] + 1]
        pers = []
        for ref_, pred_ in zip(refs, preds):
            pers.append(word_error_rate(hypotheses=[pred_], references=[ref_], use_cer=True))

        min_idx = pers.index(min(pers))
        phon_amb_indices.pop(min_idx)
        lines_to_drop.extend(phon_amb_indices)

    # drop duplicated examples, only keep with min PER
    all_preds = [x for i, x in enumerate(all_preds) if i not in lines_to_drop]
    all_references = [x for i, x in enumerate(all_references) if i not in lines_to_drop]

    wer = word_error_rate(hypotheses=all_preds, references=all_references)
    per = word_error_rate(hypotheses=all_preds, references=all_references, use_cer=True)

    print("=" * 40)
    print(f"{manifest}: PER: {per * 100:.2f}%, WER: {wer * 100:.2f}%, lines: {len(all_references)}")
    print("=" * 40)
    return wer, per

if __name__ == '__main__':
    main()


"""
python G2P/g2p_ctc_inference.py \
pretrained_model=/mnt/sdb_4/g2p/chpts/byt5/3043007_cased_with_punct/g2p/G2PCTC/2022-06-22_00-20-21/checkpoints/G2PCTC.nemo manifest_filepath=sample.json output_file=sample_PRED.json pretrained_heteronyms_model=/mnt/sdb_4/g2p/chpts/homographs_classification/G2PClassification.nemo 


/mnt/sdb_4/g2p/data_ipa/evaluation_sets/dev_cmu_phonemes.json: WER: 82.69%, PER: 35.66%

"""
