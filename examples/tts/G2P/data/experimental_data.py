import json
import os
import string
from glob import glob
from typing import List, Optional

from data_preparation_utils import (
    IPAG2PProcessor,
    check_data,
    is_valid,
    post_process,
    remove_punctuation,
    setup_tokenizer,
)
from prepare_wiki_data import prepare_wikihomograph_data
from tqdm import tqdm

from nemo.utils import logging

# download ipa dict splits using get_open_dict_splits.sh


def _prepare_ljspeech_split(
    manifest,
    phoneme_dict,
    output_dir,
    heteronyms: str = "/home/ebakhturina/NeMo/scripts/tts_dataset_files/heteronyms-052722",
):

    num_dropped = 0
    os.makedirs(output_dir, exist_ok=True)
    manifest_out = f"{output_dir}/{os.path.basename(manifest).replace('.json', '_ipa.json')}"
    ipa_tok = setup_tokenizer(phoneme_dict=phoneme_dict, heteronyms=heteronyms)

    with open(manifest_out, "w", encoding="utf-8") as f_out, open(manifest, "r", encoding="utf-8") as f_in:
        for line in tqdm(f_in):
            line = json.loads(line)

            text = post_process(line["text"])
            if not is_valid(text):
                logging.debug(set(text).difference(set(string.ascii_letters + " " + string.punctuation)))
                num_dropped += 1
                continue
            ipa_, graphemes_ = ipa_tok(text, {})

            entry = {
                "text": ipa_,
                "text_graphemes": graphemes_,
                "original_sentence": line["text"],
                "duration": 0.001,
                "audio_filepath": "n/a",
            }
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Dropped {num_dropped} from {manifest}")


def prepare_ljspeech_data(output_dir, split, cmu_dict):
    os.makedirs(output_dir, exist_ok=True)

    lj_data = {
        "train": "/mnt/sdb/DATA/LJSpeech-1.1/nvidia_ljspeech_train.json",
        "dev": "/mnt/sdb/DATA/LJSpeech-1.1/nvidia_ljspeech_val.json",
        "test": "/mnt/sdb/DATA/LJSpeech-1.1/nvidia_ljspeech_test.json",
    }

    _prepare_ljspeech_split(lj_data[split], phoneme_dict=cmu_dict, output_dir=output_dir)


def prepare_cmu(file, output_dir, split="test"):
    os.makedirs(output_dir, exist_ok=True)

    output_file_json = f"{output_dir}/{os.path.splitext(os.path.basename(file))[0]}_cmu.json"
    output_file_tsv = f"{output_dir}/{os.path.splitext(os.path.basename(file))[0]}_cmu.tsv"
    with open(file, "r") as f_in, open(output_file_json, "w") as f_out_json, open(output_file_tsv, "w") as f_out_tsv:
        for line in f_in:

            graphemes, phonemes = line.strip().split()
            phonemes = phonemes.split(",")
            if split != "test":
                phonemes == phonemes[:1]

            if graphemes.lower() == "dr.":
                graphemes = "drive"
            if is_valid(graphemes) and not graphemes.startswith("'"):
                for ph in phonemes:
                    entry = {
                        "text": ph,
                        "text_graphemes": graphemes.lower(),
                        "duration": 0.001,
                        "audio_filepath": "n/a",
                    }
                    f_out_json.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    f_out_tsv.write(f"{graphemes.lower()}\t{phonemes}\n")


def prepare_hifi_tts(
    manifest,
    phoneme_dict,
    output_dir,
    heteronyms: str = "/home/ebakhturina/NeMo/scripts/tts_dataset_files/heteronyms-052722",
):

    ipa_tok = setup_tokenizer(phoneme_dict=phoneme_dict, heteronyms=heteronyms)

    os.makedirs(output_dir, exist_ok=True)
    with open(manifest, "r") as f_in, open(f"{output_dir}/{os.path.basename(manifest)}", "w") as f_out:
        for line in tqdm(f_in):
            line = json.loads(line)
            ipa_, graphemes_ = ipa_tok(line["text_normalized"], {})

            entry = {
                "text": ipa_,
                "text_graphemes": graphemes_,
                "original_sentence": line["text_normalized"],
                "duration": line["duration"],
                "audio_filepath": line["audio_filepath"],
            }
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")


def drop_examples(manifest, output_dir, graphemes_to_exclude: Optional[List[str]] = None):
    """
    Drop examples with graphemes in the exclude list, e.g. for training set we can't used graphems from
    dev/test cmu dicts splits. Also we need ot drop sentences with OOV symbols,
    otherwise we're going have issues with comformer tokenization.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_dropped = 0
    with open(manifest, "r") as f_in, open(f"{output_dir}/{os.path.basename(manifest)}", "w") as f_out:
        for line in f_in:
            dev_test_words_present = False
            line = json.loads(line)
            text = line["text_graphemes"]
            text = post_process(text)

            if not is_valid(text):
                num_dropped += 1
                continue

            words = remove_punctuation(text).lower().split()
            for w in words:
                if graphemes_to_exclude is not None and w in graphemes_to_exclude:
                    dev_test_words_present = True
                break
            if not dev_test_words_present:
                line["text_grapheme"] = text
                line["text"] = post_process(line["text"])
                f_out.write(json.dumps(line, ensure_ascii=False) + "\n")
            else:
                num_dropped += 1
    print(f"Dropped examples with DEV/TEST graphemes {num_dropped} from {manifest}")


if __name__ == "__main__":
    """
    Use only CMU train dict part for training datasets, all CMU entries for eval/dev sets
    """
    STRESS_SYMBOLS = ["ˈ", "ˌ"]
    POST_FIX = "normalized_4"
    VERSION = 4
    # read nemo ipa cmu dict to get the order of words
    nemo_cmu = "/home/ebakhturina/NeMo/scripts/tts_dataset_files/ipa_cmudict-0.7b_nv22.06.txt"
    nemo_cmu, _ = IPAG2PProcessor._parse_as_cmu_dict(
        phoneme_dict_path=nemo_cmu, use_stresses=True, stress_symbols=STRESS_SYMBOLS, upper=True
    )

    ipa_dicts = {
        "train": "/mnt/sdb_4/g2p/data_ipa/CharsiuG2P_data_splits/train_eng-us.tsv",
        "dev": "/mnt/sdb_4/g2p/data_ipa/CharsiuG2P_data_splits/dev_eng-us.tsv",
        "test": "/mnt/sdb_4/g2p/data_ipa/CharsiuG2P_data_splits/test_eng-us.tsv",
    }

    for split in ["train", "dev", "test"]:
        tmp_file = (
            f"/tmp/{split}.txt"  # temp file store Ch CMU dict but in nemo format "  " instead "\t" separated values
        )
        with open(tmp_file, "w") as f_out, open(ipa_dicts[split], "r") as f_in:
            for line in f_in:
                grapheme, phonemes = line.strip().split("\t")
                f_out.write(f"{grapheme.upper()}  {phonemes}\n")
        ipa_dicts[split] = tmp_file

    CharsiuG2P_cmu = {}
    for split in ["train", "dev", "test"]:
        CharsiuG2P_cmu[split] = IPAG2PProcessor._parse_as_cmu_dict(
            phoneme_dict_path=ipa_dicts[split], use_stresses=True, stress_symbols=STRESS_SYMBOLS, upper=True
        )[0]

    BASE_DIR = "/mnt/sdb_4/g2p/data_ipa"
    CMU_DICT_SPLITS_DIR = f"{BASE_DIR}/nemo_cmu_splits"
    os.makedirs(CMU_DICT_SPLITS_DIR, exist_ok=True)
    # create a train.txt CMU file that contains all CharsiuG2P_cmu["train"] entries but NeMo default order for multiple entries
    # create train/dev/test dict
    with open(f"{CMU_DICT_SPLITS_DIR}/train.txt", "w") as train_f, open(
        f"{CMU_DICT_SPLITS_DIR}/dev.txt", "w"
    ) as dev_f, open(f"{CMU_DICT_SPLITS_DIR}/test.txt", "w") as test_f:
        for split in ["train", "test", "dev"]:
            for grapheme, phonemes in CharsiuG2P_cmu[split].items():
                phonemes = ["".join(x) for x in phonemes][0].split(",")

                if len(phonemes) > 1 and grapheme.upper() in nemo_cmu:
                    # check if the word is present in nemo dict, if so, use NeMo's default form
                    phonemes = ["".join(x) for x in nemo_cmu[grapheme.upper()]][0]
                else:
                    phonemes = phonemes[0]

                if split == "dev":
                    f = dev_f
                elif split == "test":
                    f = test_f
                else:
                    f = train_f

                f.write(f"{grapheme}  {phonemes}\n")

    TRAINING_DATA_DIR = f"{BASE_DIR}/training_data_v{VERSION}/raw_files"
    EVAL_DATA_DIR = f"{BASE_DIR}/evaluation_sets"
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)

    # PREPARE CMU DATA
    prepare_cmu(f"{CMU_DICT_SPLITS_DIR}/dev.txt", EVAL_DATA_DIR, "dev")
    prepare_cmu(f"{CMU_DICT_SPLITS_DIR}/test.txt", EVAL_DATA_DIR, "test")
    prepare_cmu(f"{CMU_DICT_SPLITS_DIR}/train.txt", TRAINING_DATA_DIR, "train")

    # PREPARE LJSPEECH DATA
    train_cmu_dict = "/mnt/sdb_4/g2p/data_ipa/nemo_cmu_splits/train.txt"
    complete_nemo_ipa_cmu = "/home/ebakhturina/NeMo/scripts/tts_dataset_files/ipa_cmudict-0.7b_nv22.06.txt"

    prepare_ljspeech_data(TRAINING_DATA_DIR, split="train", cmu_dict=train_cmu_dict)
    prepare_ljspeech_data(EVAL_DATA_DIR, split="dev", cmu_dict=complete_nemo_ipa_cmu)
    prepare_ljspeech_data(EVAL_DATA_DIR, split="test", cmu_dict=complete_nemo_ipa_cmu)

    # REMOVE GRAPHEMES FROM DEV AND TEST PHONEME DICTS FROM LIBRISPEECH AND WIKI DATA
    librispeech_train_manifest = (
        "/mnt/sdb_4/g2p/data_ipa/with_unicode_token/phoneme_train_all_fields_updated_word_boundaries_ipa.json"
    )

    DEV_TEST_GRAPHEMES = []
    for split in ["dev", "test"]:
        for grapheme in CharsiuG2P_cmu["dev"].keys():
            DEV_TEST_GRAPHEMES.append(grapheme.lower())

    drop_examples(librispeech_train_manifest, output_dir=TRAINING_DATA_DIR, graphemes_to_exclude=DEV_TEST_GRAPHEMES)

    # PREPARE WIKIHOMOGRAPH DATA
    prepare_wikihomograph_data(POST_FIX, output_dir=TRAINING_DATA_DIR, split="train", phoneme_dict=train_cmu_dict)
    prepare_wikihomograph_data(POST_FIX, output_dir=EVAL_DATA_DIR, split="eval", phoneme_dict=complete_nemo_ipa_cmu)
    wiki_train_manifest = f"{TRAINING_DATA_DIR}/train_wikihomograph.json"
    drop_examples(wiki_train_manifest, output_dir=TRAINING_DATA_DIR, graphemes_to_exclude=DEV_TEST_GRAPHEMES)
    wiki_eval_manifest = f"{EVAL_DATA_DIR}/eval_wikihomograph.json"
    drop_examples(wiki_eval_manifest, output_dir=EVAL_DATA_DIR, graphemes_to_exclude=None)

    # PREPARE HIFITTS DATA
    prepare_hifi_tts(f"{BASE_DIR}/all_hifi_tts.json", output_dir=TRAINING_DATA_DIR, phoneme_dict=train_cmu_dict)

    for file in glob(f"{TRAINING_DATA_DIR}/*.json"):
        check_data(file)
