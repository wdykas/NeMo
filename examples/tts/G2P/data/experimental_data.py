import json
import os
import string
from typing import List

from prepare_data import IPAG2PProcessor, remove_punctuation, setup_tokenizer
from tqdm import tqdm

# download ipa dict splits using get_open_dict_splits.sh


def is_valid(text):
    return len(set(text).difference(set(string.ascii_letters + "'"))) == 0


def _prepare_ljspeech_split(
    manifest,
    phoneme_dict,
    output_dir,
    heteronyms: str = "/home/ebakhturina/NeMo/scripts/tts_dataset_files/heteronyms-052722",
):

    os.makedirs(output_dir, exist_ok=True)
    train_manifest = f"{output_dir}/{os.path.basename(manifest).replace('.json', '_ipa.json')}"
    ipa_tok = setup_tokenizer(phoneme_dict=phoneme_dict, heteronyms=heteronyms)

    with open(train_manifest, "w", encoding="utf-8") as f_out, open(manifest, "r", encoding="utf-8") as f_in:
        for line in tqdm(f_in):
            line = json.loads(line)

            if not is_valid(line["text"]):
                continue
            ipa_, graphemes_ = ipa_tok(line["text"], {})

            entry = {
                "text": ipa_,
                "text_graphemes": graphemes_,
                "original_sentence": line["text"],
                "duration": 0.001,
                "audio_filepath": "n/a",
            }
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")


def prepare_ljspeech_data(output_dir, split, cmu_dict):
    os.makedirs(output_dir, exist_ok=True)

    lj_data = {
        "train": "/mnt/sdb/DATA/LJSpeech-1.1/nvidia_ljspeech_train.json",
        "dev": "/mnt/sdb/DATA/LJSpeech-1.1/nvidia_ljspeech_val.json",
        "test": "/mnt/sdb/DATA/LJSpeech-1.1/nvidia_ljspeech_test.json",
    }

    if split == "train":
        # convert LJSpeech train split to ipa format using train subset of open_dict
        _prepare_ljspeech_split(lj_data["train"], phoneme_dict=cmu_dict, output_dir=output_dir)

    # process dev/test sets and use all CMU dict
    if split in ["dev", "test"]:
        _prepare_ljspeech_split(lj_data[split], phoneme_dict=complete_nemo_ipa_cmu, output_dir=output_dir)


def prepare_cmu(file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(file, "r") as f_in, open(
        f"{output_dir}/{os.path.splitext(os.path.basename(file))[0]}_cmu.json", "w"
    ) as f_out:
        for line in f_in:
            graphemes, phonemes = line.strip().split()
            phonemes = phonemes.split(",")[0]
            entry = {
                "text": phonemes,
                "text_graphemes": graphemes.lower(),
                "duration": 0.001,
                "audio_filepath": "n/a",
            }
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")


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


def prepare_librispeech_data(manifest, output_dir, graphemes_to_exclude: List[str]):
    os.makedirs(output_dir, exist_ok=True)
    num_dropped = 0
    with open(manifest, "r") as f_in, open(f"{output_dir}/{os.path.basename(manifest)}", "w") as f_out:
        for line in f_in:
            dev_test_words_present = False
            text = json.loads(line)["text_graphemes"]
            words = remove_punctuation(text).lower().split()
            for w in words:
                if w in graphemes_to_exclude:
                    dev_test_words_present = True
                break
            if not dev_test_words_present:
                f_out.write(line)
            else:
                num_dropped += 1
    print(f"Dropped {num_dropped} from {manifest}")


if __name__ == "__main__":
    """
    Use only CMU train dict part for training datasets, all CMU entries for eval/dev sets
    """

    # read nemo ipa cmu dict to get the order of words
    nemo_cmu = "/home/ebakhturina/NeMo/scripts/tts_dataset_files/ipa_cmudict-0.7b_nv22.06.txt"
    nemo_cmu, _ = IPAG2PProcessor._parse_as_cmu_dict(phoneme_dict_path=nemo_cmu)

    ipa_dicts = {
        "train": "/mnt/sdb_4/g2p/data_ipa/CharsiuG2P_data_splits/train_eng-us.tsv",
        "dev": "/mnt/sdb_4/g2p/data_ipa/CharsiuG2P_data_splits/dev_eng-us.tsv",
        "test": "/mnt/sdb_4/g2p/data_ipa/CharsiuG2P_data_splits/test_eng-us.tsv",
    }

    # read dev and test graphemes for CharsiuG2P,
    # we're going to use NeMo CMU phonemes since they have proper defaults for ambiguous cases
    eval_graphemes = {}

    for split in ["dev", "test"]:
        with open(ipa_dicts[split], "r") as f_in:
            eval_graphemes[split] = []
            for line in f_in:
                grapheme, _ = line.strip().split("\t")
                eval_graphemes[split].append(grapheme.upper())
    import pdb

    pdb.set_trace()
    BASE_DIR = "/mnt/sdb_4/g2p/data_ipa"
    CMU_DICT_SPLITS_DIR = f"{BASE_DIR}/nemo_cmu_splits"
    os.makedirs(CMU_DICT_SPLITS_DIR, exist_ok=True)
    # create train/dev/test dict
    with open(f"{CMU_DICT_SPLITS_DIR}/train.txt", "w") as train_f, open(
        f"{CMU_DICT_SPLITS_DIR}/dev.txt", "w"
    ) as dev_f, open(f"{CMU_DICT_SPLITS_DIR}/test.txt", "w") as test_f:
        for grapheme, phonemes in nemo_cmu.items():
            phonemes = ["".join(x) for x in nemo_cmu[grapheme.upper()]]
            if grapheme in eval_graphemes["dev"]:
                f = dev_f
            elif grapheme in eval_graphemes["test"]:
                f = test_f
            else:
                f = train_f
            for ph in phonemes:
                f.write(f"{grapheme}  {ph}\n")

    TRAINING_DATA_DIR = f"{BASE_DIR}/training_data_v1"
    EVAL_DATA_DIR = f"{BASE_DIR}/evaluation_sets"

    # PREPARE LJSPEECH DATA
    train_cmu_dict = "/mnt/sdb_4/g2p/data_ipa/nemo_cmu_splits/train.txt"
    complete_nemo_ipa_cmu = "/home/ebakhturina/NeMo/scripts/tts_dataset_files/ipa_cmudict-0.7b_nv22.06.txt"
    prepare_ljspeech_data(TRAINING_DATA_DIR, split="train", cmu_dict=f"{CMU_DICT_SPLITS_DIR}/train.txt")
    prepare_ljspeech_data(TRAINING_DATA_DIR, split="dev", cmu_dict=complete_nemo_ipa_cmu)
    prepare_ljspeech_data(TRAINING_DATA_DIR, split="test", cmu_dict=complete_nemo_ipa_cmu)

    librispeech_train_manifest = (
        "/mnt/sdb_4/g2p/data_ipa/with_unicode_token/phoneme_train_all_fields_updated_word_boundaries_ipa.json"
    )
    prepare_librispeech_data(
        librispeech_train_manifest,
        output_dir=TRAINING_DATA_DIR,
        graphemes_to_exclude=[x.lower() for x in eval_graphemes["dev"] + eval_graphemes["test"]],
    )

    # PREPARE HIFITTS DATA
    prepare_hifi_tts(
        f"{BASE_DIR}/all_hifi_tts.json", output_dir=TRAINING_DATA_DIR, phoneme_dict=f"{CMU_DICT_SPLITS_DIR}/train.txt"
    )

    # PREPARE CMU DATA
    prepare_cmu(f"{CMU_DICT_SPLITS_DIR}/dev.txt", EVAL_DATA_DIR)
    prepare_cmu(f"{CMU_DICT_SPLITS_DIR}/test.txt", EVAL_DATA_DIR)
    prepare_cmu(f"{CMU_DICT_SPLITS_DIR}/train.txt", TRAINING_DATA_DIR)
