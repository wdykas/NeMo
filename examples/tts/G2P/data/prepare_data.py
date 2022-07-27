import csv
import json
import os
from glob import glob
from typing import List

from data_preparation_utils import get_wordid_to_nemo_cmu, is_valid, post_process, setup_tokenizer
from nemo_text_processing.g2p.data.data_utils import remove_punctuation
from tqdm import tqdm


def read_wikihomograph_normalized_file(file: str) -> (List[str], List[List[int]], List[str], List[str]):
    """
    Reads .tsv file from WikiHomograph dataset,
    e.g. https://github.com/google-research-datasets/WikipediaHomographData/blob/master/data/eval/live.tsv

    Args:
        file: path to .tsv file
    Returns:
        sentences: Text.
        start_end_indices: Start and end indices of the homograph in the sentence.
        homographs: Target homographs for each sentence (TODO: check that multiple homograph for sent are supported).
        word_ids: Word_ids corresponding to each homograph, i.e. label.
    """
    normalized_sentences = []
    word_ids = []
    with open(file, "r", encoding="utf-8") as f:
        tsv_file = csv.reader(f, delimiter="\t")
        for i, line in enumerate(tsv_file):
            if i == 0:
                continue
            _, wordid, _, sentence_normalized = line

            sentence_normalized = post_process(sentence_normalized)
            if not is_valid(sentence_normalized):
                print(sentence_normalized)
                continue

            normalized_sentences.append(sentence_normalized)
            word_ids.append(wordid)
    return normalized_sentences, word_ids


def prepare_wiki_data(
    do_lower: bool = False,
    add_punct: bool = True,
    splits: List[str] = ["train", "eval"],
    phoneme_dict: str = "/home/ebakhturina/NeMo/scripts/tts_dataset_files/ipa_cmudict-0.7b_nv22.06.txt",
    heteronyms: str = "/home/ebakhturina/NeMo/scripts/tts_dataset_files/heteronyms-052722",
):

    # to replace heteronyms with correct IPA form
    wordid_to_nemo_cmu = get_wordid_to_nemo_cmu()

    ipa_tok = setup_tokenizer(phoneme_dict=phoneme_dict, heteronyms=heteronyms)

    for subset in splits:
        # normalized data and saves in "WikipediaHomographData-master/data/{subset}_normalized"
        # is output file is present, skips normalization
        # normalize_wikihomograph_data(subset)

        normalized_data = f"/home/ebakhturina/g2p_scripts/WikipediaHomographData-master/data/{subset}_normalized/"
        files = glob(f"{normalized_data}/*.tsv")

        dir_name = f"/mnt/sdb_4/g2p/data_ipa/with_unicode_token/lower_{do_lower}_{add_punct}punct"
        os.makedirs(dir_name, exist_ok=True)
        manifest = f"{dir_name}/{subset}_wikihomograph.json"
        with open(manifest, "w", encoding="utf-8") as f_out:
            for file in tqdm(files):
                sentences, word_ids = read_wikihomograph_normalized_file(file)
                for word_id, sent in zip(word_ids, sentences):
                    sent = sent.replace(word_id, f"|{word_id}|")
                    ipa_, graphemes_ = ipa_tok(sent, wordid_to_nemo_cmu)
                    if do_lower:
                        graphemes_ = graphemes_.lower()

                    if not add_punct:
                        ipa_ = remove_punctuation(ipa_)
                        graphemes_ = remove_punctuation(graphemes_)

                    entry = {
                        "text": ipa_,
                        "text_graphemes": graphemes_,
                        "original_sentence": sent,
                        "wordid": word_id,
                        "duration": 0.001,
                        "audio_filepath": "n/a",
                    }
                    f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Data for {subset.upper()} saved at {manifest}")


def run_test():
    wordid_to_nemo_cmu = get_wordid_to_nemo_cmu()
    ipa_tok = setup_tokenizer()

    text = "How to |addict_vrb| hjls the Contents?"
    gt_ipa = "ˈhaʊ ˈtu əˈdɪkt ҂ ðə ˈkɑntɛnts?"
    gt_graphemes = "How to addict ҂ the Contents?"
    pred_phons, pred_graphemes = ipa_tok(text, wordid_to_nemo_cmu)
    assert pred_phons == gt_ipa and pred_graphemes == gt_graphemes

    text = "hello-World-waveglow!"
    gt_ipa = "həˈɫoʊ-ˈwɝɫd-ˈweɪvˌɡɫoʊ!"
    gt_graphemes = "hello-World-waveglow!"
    pred_phons, pred_graphemes = ipa_tok(text, wordid_to_nemo_cmu)
    assert pred_phons == gt_ipa and pred_graphemes == gt_graphemes

    text = "It has used other Treasury law"
    pred_phons, pred_graphemes = ipa_tok(text, {})
    print(pred_phons)
    print(pred_graphemes)
    import pdb

    pdb.set_trace()
    print()


if __name__ == "__main__":
    run_test()
