import json
import re
from typing import List, Optional

from nemo_text_processing.text_normalization.normalize import Normalizer
from tqdm import tqdm

from nemo.collections.tts.torch.en_utils import english_word_tokenize
from nemo.collections.tts.torch.g2ps import IPAG2P


class IPAG2PProcessor(IPAG2P):
    def __init__(
        self,
        phoneme_dict,
        word_tokenize_func=english_word_tokenize,
        ignore_ambiguous_words=True,
        heteronyms=None,
        grapheme_unk_token="<unk>",
        ipa_unk_token="§",
        phoneme_probability: Optional[float] = None,
        use_stresses: Optional[bool] = True,
        set_graphemes_upper: Optional[bool] = True,
    ):

        self.grapheme_unk_token = grapheme_unk_token
        self.ipa_unk_token = ipa_unk_token

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
                return self.phoneme_dict[word[:-2]][0] + ["t", "s"], True
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
            if word[-3] == 'T':
                # Case like "airport's"
                return self.phoneme_dict[word[:-2]][0] + ["t", "s"], True
            elif word[-3] == 'S':
                # Case like "jones's"
                return self.phoneme_dict[word[:-2]][0] + ["ɪ", "z"], True
            else:
                return self.phoneme_dict[word[:-2]][0] + ["z"], True

        # Phoneme dict lookup for unique words (or default pron if ignore_ambiguous_words=False)
        if word in self.phoneme_dict and (not self.ignore_ambiguous_words or self.is_unique_in_phoneme_dict(word)):
            return self.phoneme_dict[word][0], True

        # use ipa_unk_token for OOV words
        return self.apply_to_oov_word(word), False

    def __call__(self, text):
        words = self.word_tokenize_func(text)

        prons = []
        graphemes = []
        for word, without_changes in words:
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


if __name__ == "__main__":

    from nemo.collections.common.tokenizers.char_tokenizer import CharTokenizer
    import string

    def setup_grapheme_tokenizer(grapheme_unk_token="Ӿ"):
        punctuation_marks = string.punctuation.replace('"', "").replace("\\", "")
        chars = string.ascii_lowercase + string.ascii_uppercase + punctuation_marks + grapheme_unk_token + " "
        vocab_file = "/tmp/char_vocab.txt"
        with open(vocab_file, "w") as f:
            [f.write(f'"{ch}"\n') for ch in chars]
            f.write('"\\""\n')  # add " to the vocab

        return CharTokenizer(vocab_file=vocab_file, unk_token=grapheme_unk_token)

    tokenizer_grapheme = setup_grapheme_tokenizer()

    print(tokenizer_grapheme.text_to_tokens('hello, Ӿ "world"! and ҂ it?'))
    print(tokenizer_grapheme.text_to_tokens("hello, Ӿ world! and ҂ it's?"))

    import pdb

    pdb.set_trace()

    # for p in PUNCT_LIST:
    #     if p not in (string.ascii_lowercase + string.ascii_uppercase):
    #         print(f"{p} is missing")
    #
    # import pdb;
    #
    # pdb.set_trace()
    from nemo.collections.tts.torch.tts_tokenizers import EnglishCharsTokenizer, BaseCharsTokenizer
    from nemo.collections.tts.torch.en_utils import english_text_preprocessing
    import string

    class EnglishCharsTokenizerForG2P(BaseCharsTokenizer):
        def __init__(
            self,
            punct=True,
            apostrophe=True,
            add_blank_at=None,
            pad_with_space=False,
            non_default_punct_list=None,
            grapheme_unk_token="҂",
            text_preprocessing_func=english_text_preprocessing,
        ):
            """English char-based tokenizer.
            Args:
                punct: Whether to reserve grapheme for basic punctuation or not.
                apostrophe: Whether to use apostrophe or not.
                add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
                 if None then no blank in labels.
                pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
                non_default_punct_list: List of punctuation marks which will be used instead default.
                text_preprocessing_func: Text preprocessing function for correct execution of the tokenizer.
                 Basically, it replaces all non-unicode characters with unicode ones and apply lower() function.
            """
            super().__init__(
                chars=string.ascii_lowercase + string.ascii_uppercase + grapheme_unk_token,
                punct=punct,
                apostrophe=apostrophe,
                add_blank_at=add_blank_at,
                pad_with_space=pad_with_space,
                non_default_punct_list=non_default_punct_list,
                text_preprocessing_func=text_preprocessing_func,
            )

    eng_char_tokenizer = EnglishCharsTokenizerForG2P(
        punct=True, apostrophe=True, add_blank_at=None, pad_with_space=False, non_default_punct_list=None
    )

    # def replace_oov_token_in_eng_char_tokenizer(eng_char_tokenizer, new_oov_token="҂"):
    #     """
    #     Replace tokenizer's OOV token to match T5 model token
    #     :param eng_char_tokenizer:
    #     :param new_oov_token:
    #     :return:
    #     """
    #     old_oov_token_id = eng_char_tokenizer.oov
    #     old_oov_token = eng_char_tokenizer.tokens[old_oov_token_id]
    #
    #     del eng_char_tokenizer._token2id[old_oov_token]
    #     eng_char_tokenizer._token2id[new_oov_token] = old_oov_token_id
    #     eng_char_tokenizer.tokens[old_oov_token_id] = new_oov_token
    #     eng_char_tokenizer._id2token[old_oov_token_id] = new_oov_token
    #
    #     return eng_char_tokenizer

    print("BEFORE:", eng_char_tokenizer("Hello, world! ҂"))

    import pdb

    pdb.set_trace()
    ipa_tok = IPAG2PProcessor(
        phoneme_dict="/home/ebakhturina/NeMo/scripts/tts_dataset_files/ipa_cmudict-0.7b_nv22.06.txt",
        heteronyms="/home/ebakhturina/NeMo/scripts/tts_dataset_files/heteronyms-052722",
        grapheme_unk_token="<unk>",
        ipa_unk_token="§",
        ignore_ambiguous_words=False,
        set_graphemes_upper=False,
        use_stresses=True,
    )

    text = "how to use the contents?"
    gt_ipa = "ˈhaʊ ˈtu § ðə ˈkɑntɛnts?"
    gt_graphemes = "how to <unk> the contents?"
    pred_phons, pred_graphemes = ipa_tok(text)
    assert pred_phons == gt_ipa and pred_graphemes == gt_graphemes

    text = ipa_tok("hello-world-waveglow!")
    gt_ipa = "həˈɫoʊ-ˈwɝɫd-ˈweɪvˌɡɫoʊ!"
    gt_graphemes = "hello-world-waveglow!"
    pred_phons, pred_graphemes = ipa_tok(text)
    assert pred_phons == gt_ipa and pred_graphemes == gt_graphemes

    with open("test.json", "r") as f:
        for line in f:
            line = json.loads(line)
            text = line["text_graphemes"]
            ipa = line["text"]
            prons, graphemes = ipa_tok(text)
            print("=" * 40)
            print(f"GT   -> {text}")
            print(f"PRED -> {graphemes}\n")
            print(f"IPA GT   -> {ipa}")
            print(f"IPA PRED -> {prons}")

    print("".join(ipa_tok("Hello, world!")))
