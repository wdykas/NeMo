import json
import os
from typing import List

import torch
from transformers import PreTrainedTokenizerBase

from nemo.core import Dataset
from nemo.utils import logging


class CTCG2PBPEDataset(Dataset):
    """
    Creates a dataset to train a T5G2P model.
    """

    def __init__(
        self,
        manifest_filepath: str,
        tokenizer_graphemes: PreTrainedTokenizerBase,
        tokenizer_phonemes: PreTrainedTokenizerBase,
        do_lower: bool = True,
        labels: List[str] = None,
        max_source_len: int = 512,
        max_target_len: int = 512,
        with_labels: bool = True,
    ):
        # TODO: docstring
        super().__init__()

        if not os.path.exists(manifest_filepath):
            raise ValueError(f"{manifest_filepath} not found")

        self.manifest = manifest_filepath
        self.tokenizer_graphemes = tokenizer_graphemes
        self.tokenizer_phonemes = tokenizer_phonemes
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.labels = labels
        self.labels_tkn2id = {l: i for i, l in enumerate(labels)}
        self.data = []
        self.pad_token = 0
        self.with_labels = with_labels

        removed_ctc_max = 0
        removed_source_max = 0
        with open(manifest_filepath, 'r') as f_in:
            logging.debug(f"Loading dataset from: {manifest_filepath}")
            for i, line in enumerate(f_in):
                item = json.loads(line)

                # if len(item["text"]) > max_source_len:
                #     num_filtered += 1
                #     continue
                # if len(item["pred_text"]) > max_target_len:
                #     num_filtered += 1
                #     continue

                if do_lower:
                    item["text_graphemes"] = item["text_graphemes"].lower()

                if isinstance(self.tokenizer_graphemes, PreTrainedTokenizerBase):
                    grapheme_tokens = self.tokenizer_graphemes(item["text_graphemes"])
                    grapheme_tokens_len = len(grapheme_tokens['input_ids'])
                else:
                    grapheme_tokens = self.tokenizer_graphemes.text_to_ids(item["text_graphemes"])
                    grapheme_tokens_len = len(grapheme_tokens)

                if with_labels:
                    target_tokens = self.tokenizer_phonemes.text_to_ids(item["text"])
                    target_len = len(target_tokens)

                    if target_len > grapheme_tokens_len:
                        removed_ctc_max += 1
                        # print(f"CTC: target: {target_len} -- input: {grapheme_tokens_len} -- {item['text_graphemes']} -- {item['text']}")
                        # if target_len - grapheme_tokens_len > 4:
                        #     import pdb; pdb.set_trace()
                        #     print()

                        # seq_lengths.append(len(item["text_graphemes"]))
                        # sentences.append(item["text_graphemes"])
                        continue

                    if grapheme_tokens_len > max_source_len:
                        removed_source_max += 1
                        # seq_lengths.append(len(item["text_graphemes"]))
                        # sentences.append(item["text_graphemes"])
                        continue

                    self.data.append(
                        {
                            "graphemes": item["text_graphemes"],
                            "phonemes": item["text"],
                            "target": target_tokens,
                            "target_len": target_len,
                        }
                    )
                else:
                    if len(grapheme_tokens) > max_source_len:
                        item["text_graphemes"] = item["text_graphemes"][:max_source_len]
                        removed_source_max += 1
                    self.data.append(
                        {"graphemes": item["text_graphemes"],}
                    )

        logging.info(
            f"REMOVED based on CTC max: {removed_ctc_max} examples, based on MAX_SOURCE_LEN: {removed_source_max} examples from {manifest_filepath}"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def map(self, text: str) -> List[int]:
        """ Creates a mapping from target labels to ids."""
        tokens = []
        for word_id, word in enumerate(text.split()):
            tokens.append(self.labels_tkn2id[word])
        return tokens

    def _collate_fn(self, batch):
        """
        from nemo.collections.common.tokenizers import TokenizerSpec
        if isinstance(self.tokenizer, TokenizerSpec):
            input_ids, attention_mask = self.__get_input_ids(graphemes_batch, pad_token=0, return_mask=True)
            labels = self.__get_input_ids(graphemes_batch, pad_token=-100, return_mask=False)
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)
            labels = torch.tensor(labels)
        else:
            # Encode inputs (graphemes)
            input_encoding = self.tokenizer(
                graphemes_batch, padding='longest', max_length=self.max_source_len, truncation=True, return_tensors='pt',
            )
            input_ids, attention_mask = input_encoding.input_ids, input_encoding.attention_mask

            # Encode targets (phonemes)
            target_encoding = self.tokenizer(
                phonemes_batch, padding='longest', max_length=self.max_target_len, truncation=True,
            )
            labels = target_encoding.input_ids

            # Need to replace padding tokens w/ -100 for loss to ignore them
            labels = [
                [(label if label != self.tokenizer.pad_token_id else -100) for label in labels_example]
                for labels_example in labels
            ]
            labels = torch.tensor(labels)

        :param batch:
        :return:
        """
        graphemes_batch = [entry["graphemes"] for entry in batch]

        # Encode inputs (graphemes)
        if isinstance(self.tokenizer_graphemes, PreTrainedTokenizerBase):
            input_encoding = self.tokenizer_graphemes(
                graphemes_batch,
                padding='longest',
                max_length=self.max_source_len,
                truncation=True,
                return_tensors='pt',
            )
            input_ids, attention_mask = input_encoding.input_ids, input_encoding.attention_mask
            input_len = torch.sum(attention_mask, 1) - 1
        else:
            input_ids = [self.tokenizer_graphemes.text_to_ids(sentence) for sentence in graphemes_batch]
            input_len = [len(entry) for entry in input_ids]
            max_len = max(input_len)
            input_ids = [entry + [0] * (max_len - entry_len) for entry, entry_len in zip(input_ids, input_len)]
            attention_mask = None  # not used with Conformer encoder
            input_ids = torch.tensor(input_ids)
            input_len = torch.tensor(input_len)

        # inference
        if not self.with_labels:
            output = (input_ids, attention_mask, input_len)
        # Encode targets (phonemes)
        else:
            targets = [torch.tensor(entry["target"]) for entry in batch]
            target_lengths = [torch.tensor(entry["target_len"]) for entry in batch]
            max_target_len = max(target_lengths)

            padded_targets = []
            for target, target_len in zip(targets, target_lengths):
                pad = (0, max_target_len - target_len)
                target_pad = torch.nn.functional.pad(target, pad, value=len(self.labels))
                padded_targets.append(target_pad)

            padded_targets = torch.stack(padded_targets)
            target_lengths = torch.stack(target_lengths)
            output = (input_ids, attention_mask, input_len, padded_targets, target_lengths)

        return output
