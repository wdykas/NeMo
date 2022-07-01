import json
import re
import string
from typing import List, Optional

from tqdm import tqdm

from nemo.collections.tts.torch.en_utils import english_word_tokenize
from nemo.collections.tts.torch.g2ps import IPAG2P


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


def post_process(text):
    text = text.replace('“', '"').replace('”', '"').replace("—", "-")
    return text


def remove_punctuation(text: str, exclude=None):
    all_punct_marks = string.punctuation

    if exclude is not None:
        for p in exclude:
            all_punct_marks = all_punct_marks.replace(p, "")
    text = re.sub("[" + all_punct_marks + "]", " ", text)

    text = re.sub(r" +", " ", text)
    return text.strip()


def is_valid(text, unk_token="҂", verbose=False):
    invalid_symbols = set(text).difference(set(unk_token + string.ascii_letters + " " + string.punctuation))

    if verbose and len(invalid_symbols) > 0:
        print(invalid_symbols)
    return len(invalid_symbols) == 0


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


def get_wordid_to_nemo_cmu(wordid_to_nemo_file="wordid_to_nemo_cmu.tsv"):
    # to replace heteronyms with correct IPA form
    wordid_to_nemo_cmu = {}
    with open(wordid_to_nemo_file, "r", encoding="utf-8") as f:
        for line in f:
            word_id, ipa = line.strip().split("\t")
            wordid_to_nemo_cmu[word_id] = ipa
    return wordid_to_nemo_cmu


def check_data(manifest):
    num_dropped = 0
    with open(manifest, "r", encoding="utf-8") as f_in:
        for line in tqdm(f_in):
            line = json.loads(line)

            text = line["text_graphemes"]
            if not is_valid(text, verbose=True):
                import pdb

                pdb.set_trace()

                print(line)
                num_dropped += 1
                continue
    if num_dropped > 0:
        print(f"Final check_data() dropped: {num_dropped} in {manifest}")
