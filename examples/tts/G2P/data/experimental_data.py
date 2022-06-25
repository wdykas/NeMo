import json
import os

from prepare_data import IPAG2PProcessor, setup_tokenizer
from tqdm import tqdm

# download ipa dict splits using get_open_dict_splits.sh


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
            ipa_, graphemes_ = ipa_tok(line["text"], {})

            entry = {
                "text": ipa_,
                "text_graphemes": graphemes_,
                "original_sentence": line["text"],
                "duration": 0.001,
                "audio_filepath": "n/a",
            }
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")


def prepare_ljspeech_data(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    cmu_dict_dir = "/mnt/sdb_4/g2p/data_ipa/nemo_cmu_splits"
    complete_nemo_ipa_cmu = "/home/ebakhturina/NeMo/scripts/tts_dataset_files/ipa_cmudict-0.7b_nv22.06.txt"

    lj_data = {
        "train": "/mnt/sdb/DATA/LJSpeech-1.1/nvidia_ljspeech_train.json",
        "dev": "/mnt/sdb/DATA/LJSpeech-1.1/nvidia_ljspeech_val.json",
        "test": "/mnt/sdb/DATA/LJSpeech-1.1/nvidia_ljspeech_test.json",
    }

    # convert LJSpeech train split to ipa format using train subset of open_dict
    _prepare_ljspeech_split(lj_data["train"], phoneme_dict=f"{cmu_dict_dir}/train.txt", output_dir=output_dir)

    # process dev/test sets and use all CMU dict
    for split in ["dev", "test"]:
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


if __name__ == "__main__":
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

    BASE_DIR = "/mnt/sdb_4/g2p/data_ipa"
    OUTPUT_DIR = f"{BASE_DIR}/nemo_cmu_splits"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # create train/dev/test dict
    with open(f"{OUTPUT_DIR}/train.txt", "w") as train_f, open(f"{OUTPUT_DIR}/dev.txt", "w") as dev_f, open(
        f"{OUTPUT_DIR}/test.txt", "w"
    ) as test_f:
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
    prepare_ljspeech_data()

    prepare_cmu(f"{OUTPUT_DIR}/dev.txt", EVAL_DATA_DIR)
    prepare_cmu(f"{OUTPUT_DIR}/test.txt", EVAL_DATA_DIR)
    prepare_cmu(f"{OUTPUT_DIR}/trai.txt", TRAINING_DATA_DIR)
