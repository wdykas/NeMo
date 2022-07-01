import json
import os
from glob import glob

from data_preparation_utils import check_data, get_wordid_to_nemo_cmu, is_valid, post_process, setup_tokenizer
from nemo_text_processing.text_normalization.normalize import Normalizer
from tqdm import tqdm

from nemo.collections.tts.torch.g2p_utils.data_utils import correct_wikihomograph_data, read_wikihomograph_file


def post_process_normalization(text):
    text = text.replace("slash ", "/ ").replace(" slash[", " slash [").replace("—", "-")
    return text


def normalize_wikihomograph_data(subset, post_fix):
    BASE_DIR = "/home/ebakhturina/g2p_scripts/"
    output_dir = f"{BASE_DIR}/WikipediaHomographData-master/data/{subset}_{post_fix}"
    os.makedirs(output_dir, exist_ok=True)

    normalizer = Normalizer(lang="en", input_case="cased", cache_dir="cache_dir", overwrite_cache=False)
    num_removed = 0

    for file in tqdm(glob(f"{BASE_DIR}/WikipediaHomographData-master/data/{subset}/*.tsv")):
        file_name = os.path.basename(file)
        output_f = f"{output_dir}/{file_name.replace('.tsv', '.json')}"
        if os.path.exists(output_f):
            continue

        sentences, start_end_indices, homographs, word_ids = read_wikihomograph_file(file)
        with open(output_f, "w") as f_out:
            for i, sent in enumerate(sentences):
                start, end = start_end_indices[i]
                sent, start, end = correct_wikihomograph_data(sent, start, end)
                homograph = file_name.replace(".tsv", "")

                replace_token = "[]"
                homograph_span = sent[start:end]
                if homograph_span.lower() != homograph and sent.lower().count(homograph) == 1:
                    start = sent.lower().index(homograph)
                    end = start + len(homograph)
                    homograph_span = sent[start:end].lower()
                    assert homograph == homograph_span.lower()

                # we'll skip examples where start/end indices are incorrect and
                # the target homorgaph is also present in the context (ambiguous)
                if homograph_span.lower() != homograph:
                    import pdb

                    pdb.set_trace()
                    num_removed += 1
                else:
                    sentence_to_normalize = sent[: int(start)] + replace_token + sent[int(end) :]
                    try:
                        norm_text = normalizer.normalize(
                            text=sentence_to_normalize, verbose=False, punct_post_process=True, punct_pre_process=True,
                        )
                    except:
                        print("TN ERROR: ", sentence_to_normalize)
                        num_removed += 1

                    norm_text = post_process_normalization(norm_text)
                    entry = {
                        "text_graphemes": norm_text,
                        "norm_text_graphemes": norm_text.replace(replace_token, homograph_span),
                        "start_end": [start, end],
                        "homograph_span": homograph_span,
                        "word_id": word_ids[i],
                    }
                    f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Normalized data is saved at {output_dir}, number of removed lines: {num_removed}")


def _prepare_wikihomograph_data(post_fix, output_dir, phoneme_dict, split):
    drop = []
    replace_token = "[]"
    # to replace heteronyms with correct IPA form
    wordid_to_nemo_cmu = get_wordid_to_nemo_cmu("/home/ebakhturina/NeMo/examples/tts/G2P/data/wordid_to_nemo_cmu.tsv")
    ipa_tok = setup_tokenizer(phoneme_dict=phoneme_dict)
    normalized_data = f"/home/ebakhturina/g2p_scripts/WikipediaHomographData-master/data/{split}_{post_fix}/"
    files = glob(f"{normalized_data}/*.json")

    os.makedirs(output_dir, exist_ok=True)
    manifest = f"{output_dir}/{split}_wikihomograph.json"
    with open(manifest, "w", encoding="utf-8") as f_out:
        for file in tqdm(files):
            with open(file, "r") as f_in:
                for line in f_in:
                    line = json.loads(line)
                    graphemes = line["text_graphemes"]
                    ipa_, graphemes_ = ipa_tok(graphemes, wordid_to_nemo_cmu)
                    graphemes_ = graphemes_.replace(replace_token, line["homograph_span"])
                    heteronym_ipa = wordid_to_nemo_cmu[line["word_id"]]
                    ipa_ = ipa_.replace(replace_token, heteronym_ipa)
                    graphemes_ = post_process(graphemes_)
                    if not is_valid(graphemes_):
                        drop.append(graphemes_)
                    else:
                        line["text_graphemes"] = graphemes_
                        line["text"] = post_process(ipa_)
                        line["duration"] = (0.001,)
                        line["audio_filepath"] = "n/a"
                        f_out.write(json.dumps(line, ensure_ascii=False) + "\n")
        return manifest
        print(
            f"During validation check in dataset preparation dropped: {len(drop)}, Data for {split.upper()} saved at {manifest}"
        )


def prepare_wikihomograph_data(post_fix, output_dir, split, phoneme_dict=None):
    if phoneme_dict is None:
        phoneme_dict = "/home/ebakhturina/NeMo/scripts/tts_dataset_files/ipa_cmudict-0.7b_nv22.06.txt"
    normalize_wikihomograph_data(split, post_fix)
    manifest = _prepare_wikihomograph_data(post_fix, split=split, output_dir=output_dir, phoneme_dict=phoneme_dict)
    print('checking..')
    check_data(manifest)


if __name__ == "__main__":
    split = "eval"
    post_fix = "normalized_3"
    output_dir = "TMP"
    prepare_wikihomograph_data(post_fix, output_dir=output_dir, split=split)
