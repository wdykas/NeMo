import csv
import json
import os
import re
import string
from glob import glob
from typing import List, Optional

from nemo_text_processing.text_normalization.normalize import Normalizer
from tqdm import tqdm

from nemo.collections.tts.torch.en_utils import english_word_tokenize
from nemo.collections.tts.torch.g2ps import IPAG2P

VOCAB = string.punctuation + string.ascii_letters + " "


class IPAG2PProcessor(IPAG2P):
    def __init__(
        self,
        phoneme_dict,
        word_tokenize_func=english_word_tokenize,
        do_lower=True,
        ignore_ambiguous_words=True,
        heteronyms=None,
        grapheme_unk_token="҂",
        ipa_unk_token="҂",
        phoneme_probability: Optional[float] = None,
        use_stresses: Optional[bool] = True,
        set_graphemes_upper: Optional[bool] = True,
    ):

        self.grapheme_unk_token = grapheme_unk_token
        self.ipa_unk_token = ipa_unk_token
        self.do_lower = do_lower

        super().__init__(
            phoneme_dict=phoneme_dict,
            word_tokenize_func=word_tokenize_func,
            apply_to_oov_word=lambda x: self.ipa_unk_token,
            ignore_ambiguous_words=ignore_ambiguous_words,
            heteronyms=heteronyms,
            phoneme_probability=phoneme_probability,
            use_stresses=use_stresses,
            set_graphemes_upper=set_graphemes_upper,
        )

    def post_process(self, symbols: List[str]):
        return "".join(symbols)

    def parse_one_word(self, word: str):
        """
		Returns parsed `word` and `status` (bool: False if word wasn't handled, True otherwise).
		"""
        word = word.lower()
        if self.set_graphemes_upper:
            word = word.upper()

        if self.phoneme_probability is not None and self._rng.random() > self.phoneme_probability:
            return word, True

        # Punctuation (assumes other chars have been stripped)
        if re.search("[a-zA-Z]", word) is None:
            return list(word), True

        # Heteronym
        if self.heteronyms and word in self.heteronyms:
            return self.apply_to_oov_word(word), False

        # `'s` suffix (with apostrophe) - not in phoneme dict
        if (
            len(word) > 2
            and word.endswith("'s")
            and (word not in self.phoneme_dict)
            and (word[:-2] in self.phoneme_dict)
            and (not self.ignore_ambiguous_words or self.is_unique_in_phoneme_dict(word[:-2]))
        ):
            if word[-3] == 'T':
                # Case like "airport's"
                return self.phoneme_dict[word[:-2]][0] + ["s"], True
            elif word[-3] == 'S':
                # Case like "jones's"
                return self.phoneme_dict[word[:-2]][0] + ["ɪ", "z"], True
            else:
                return self.phoneme_dict[word[:-2]][0] + ["z"], True

        # `s` suffix (without apostrophe) - not in phoneme dict
        if (
            len(word) > 1
            and word.endswith("s")
            and (word not in self.phoneme_dict)
            and (word[:-1] in self.phoneme_dict)
            and (not self.ignore_ambiguous_words or self.is_unique_in_phoneme_dict(word[:-1]))
        ):
            if word[-2] == 'T':
                # Case like "airports"
                return self.phoneme_dict[word[:-1]][0] + ["s"], True
            else:
                return self.phoneme_dict[word[:-1]][0] + ["z"], True

        # Phoneme dict lookup for unique words (or default pron if ignore_ambiguous_words=False)
        if word in self.phoneme_dict and (not self.ignore_ambiguous_words or self.is_unique_in_phoneme_dict(word)):
            return self.phoneme_dict[word][0], True

        # use ipa_unk_token for OOV words
        return self.apply_to_oov_word(word), False

    def __call__(self, text, wordid_to_cmu_dict):
        words = self.word_tokenize_func(text, lower=self.do_lower)
        prons = []
        graphemes = []
        for word, without_changes in words:
            if isinstance(word, list):
                if len(word) == 1 and word[0] in wordid_to_cmu_dict:
                    prons.append(wordid_to_cmu_dict[word[0]])
                    word = word[0].split("_")[0]
                    graphemes.append(word)
                    continue
                else:
                    raise ValueError(f"Input won't be processed: {text}")
            else:
                pron, phon_is_handled = self.parse_one_word(word)

            if phon_is_handled:
                graphemes.append(word)
            else:
                word_by_hyphen = word.split("-")
                if len(word_by_hyphen) > 1:
                    pron = []
                    for i, sub_word in enumerate(word_by_hyphen):
                        p, phon_is_handled = self.parse_one_word(sub_word)
                        pron.extend(p)
                        if phon_is_handled:
                            graphemes.append(sub_word)
                        else:
                            assert p == self.ipa_unk_token
                            graphemes.append(self.grapheme_unk_token)

                        if i < (len(word_by_hyphen) - 1):
                            pron.extend(["-"])
                            graphemes.append("-")

                else:
                    assert pron == self.ipa_unk_token
                    graphemes.append(self.grapheme_unk_token)

            prons.extend(pron)

        prons = self.post_process(prons)
        graphemes = self.post_process(graphemes)
        return prons, graphemes


def remove_punctuation(text: str, exclude=None):
    all_punct_marks = string.punctuation

    if exclude is not None:
        for p in exclude:
            all_punct_marks = all_punct_marks.replace(p, "")
    text = re.sub("[" + all_punct_marks + "]", " ", text)

    text = re.sub(r" +", " ", text)
    return text.strip()


def post_process(text):
    text = text.replace("—", "-")
    return text


def is_valid(text):
    # contains only letters and punctuation marks
    return len(set(text).difference(set(VOCAB))) == 0


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


def setup_tokenizer(
    phoneme_dict: str = "/home/ebakhturina/NeMo/scripts/tts_dataset_files/ipa_cmudict-0.7b_nv22.06.txt",
    heteronyms: str = "/home/ebakhturina/NeMo/scripts/tts_dataset_files/heteronyms-052722",
    unk_token: str = "҂",
):
    return IPAG2PProcessor(
        phoneme_dict=phoneme_dict,
        heteronyms=heteronyms,
        grapheme_unk_token=unk_token,
        ipa_unk_token=unk_token,
        do_lower=False,
        ignore_ambiguous_words=False,
        set_graphemes_upper=False,
        use_stresses=True,
    )


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


def get_wordid_to_nemo_cmu():
    # to replace heteronyms with correct IPA form
    wordid_to_nemo_cmu = {}
    with open("wordid_to_nemo_cmu.tsv", "r", encoding="utf-8") as f:
        for line in f:
            word_id, ipa = line.strip().split("\t")
            wordid_to_nemo_cmu[word_id] = ipa
    return wordid_to_nemo_cmu


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
