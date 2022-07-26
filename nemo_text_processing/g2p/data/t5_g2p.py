# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import torch
from transformers import PreTrainedTokenizerBase

from nemo.core.classes import Dataset
from nemo.utils import logging


class T5G2PDataset(Dataset):
    """
    Creates a dataset to train a T5G2P model.
    """

    def __init__(
        self,
        manifest_filepath: str,
        tokenizer: PreTrainedTokenizerBase,
        max_source_len: int = 512,
        max_target_len: int = 512,
        do_lower: bool = False,
        grapheme_field: str = "text_graphemes",
        phoneme_field: str = "text",
    ):
        """
        Dataset to train T5-based G2P models.

        Args:
            manifest_filepath: path to a .json manifest that contains "phoneme_field" and "grapheme_field"
            tokenizer: pretrained T5 tokenizer
            max_source_len: max length of the grapheme input sequence (examples exceeding len will be dropped)
            max_target_len: max length of the phoneme sequence (examples exceeding len will be dropped)
            do_lower: a flag that indicates whether to lower case input grapheme sequence
            phoneme_field: name of the field in manifest_filepath for ground truth phonemes
            grapheme_field: name of the field in manifest_filepath for input grapheme text
        """
        super().__init__()

        if not os.path.exists(manifest_filepath):
            raise ValueError(f"{manifest_filepath} not found")

        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.do_lower = do_lower
        self.data = []

        num_filtered = 0

        # Load grapheme/phoneme sequence pairs into self.data
        with open(manifest_filepath, 'r') as f_in:
            logging.info(f"Loading dataset from: {manifest_filepath}")
            for line in f_in:
                item = json.loads(line)

                if len(item[grapheme_field]) > max_source_len:
                    num_filtered += 1
                    logging.debug(f"dropping {len(item[grapheme_field])} longer max_source_len")
                    continue
                if len(item[phoneme_field]) > max_target_len:
                    num_filtered += 1
                    logging.debug(f"dropping {len(item[phoneme_field])} longer max_target_len")
                    continue

                graphemes = item[grapheme_field]
                if do_lower:
                    graphemes = graphemes.lower()
                self.data.append({"graphemes": graphemes, "phonemes": item[phoneme_field]})

        logging.info(f"Filtered {num_filtered} entries.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def _collate_fn(self, batch):
        graphemes_batch = [entry["graphemes"] for entry in batch]
        phonemes_batch = [entry["phonemes"] for entry in batch]
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

        return (input_ids, attention_mask, labels)  # grapheme IDs, attention mask, phoneme IDs
