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


import csv
import os
from collections import defaultdict
from glob import glob
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo.collections.tts.torch.g2p_utils.data_utils import (
    correct_wikihomograph_data,
    read_wikihomograph_file,
    read_wordids,
)
import sys
sys.path.append("/home/ebakhturina/NeMo/nemo/collections/tts/torch/g2p_utils")
from convert_aligner_data_to_g2p_classification import convert_to_wiki_format
from nemo.core.classes import Dataset
from nemo.utils import logging

__all__ = ['G2PClassificationDataset']


class G2PClassificationDataset(Dataset):
    """
    Creates dataset to use to train a DuplexDecoderModel.
    Converts from raw data to an instance that can be used by Dataloader.
    For dataset to use to do end-to-end inference, see TextNormalizationTestDataset.

    Args:
        # input_file: path to the raw data file (e.g., train.tsv).
        # 	For more info about the data format, refer to the
        # 	`text_normalization doc <https://github.com/NVIDIA/NeMo/blob/main/docs/source/nlp/text_normalization.rst>`.
        # raw_instances: processed raw instances in the Google TN dataset format (used for tarred dataset)
        # tokenizer: tokenizer of the model that will be trained on the dataset
        # tokenizer_name: name of the tokenizer,
        # mode: should be one of the values ['tn', 'itn', 'joint'].  `tn` mode is for TN only.
        # 	`itn` mode is for ITN only. `joint` is for training a system that can do both TN and ITN at the same time.
        # max_len: maximum length of sequence in tokens. The code will discard any training instance whose input or
        # 	output is longer than the specified max_len.
        # decoder_data_augmentation (bool): a flag indicates whether to augment the dataset with additional data
        # 	instances that may help the decoder become more robust against the tagger's errors.
        # 	Refer to the doc for more info.
        # lang: language of the dataset
        # use_cache: Enables caching to use pickle format to store and read data from
        # max_insts: Maximum number of instances (-1 means no limit)
        # do_tokenize: Tokenize each instance (set to False for Tarred dataset)
        # initial_shuffle: Set to True to shuffle the data
    """

    def __init__(
        self,
        dir_name: str,
        tokenizer: PreTrainedTokenizerBase,
        wordid_map: str = "wordids.tsv",
        context_len: int = 3,
        max_seq_len: int = 512,
        with_labels: bool = True,
    ):
        """
        # TODO: docstring
        :type input_file: object
        """

        super().__init__()

        if not os.path.exists(dir_name):
            raise ValueError(f"{dir_name} not found")

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = []
        self.pad_token = 0
        self.with_labels = with_labels

        self.wiki_homograph_dict, self.target_ipa, self.target_ipa_label_to_id = read_wordids(wordid_map)

        for file in tqdm(glob(f"{dir_name}/*.tsv")):
            file_data = read_wikihomograph_file(file)

            for sentence, start_end_index, homograph, word_id in zip(*file_data):
                start, end = start_end_index
                target, target_and_negatives = self._prepare_sample(sentence, start, end, homograph, word_id)
                self.data.append({"input": sentence, "target": target, "target_and_negatives": target_and_negatives})
        print("--->", len(self.data), dir_name)

        # TODO: refactor
        if "train" in dir_name:
            #  add Aligner data data
            lj_manifest = "/mnt/sdb_4/g2p/data_ipa/training_data_v8/raw_files/filtered_disamb_ljspeech_train_ipa.json"
            hifi_9017_manifest = "/mnt/sdb_4/g2p/data_ipa/training_data_v8/raw_files/disamb_9017_clean_train_heteronyms_filtered_ipa.json"
            data = convert_to_wiki_format([lj_manifest, hifi_9017_manifest])
            for sentence, start_end_index, homograph, word_id in zip(*data):
                start, end = start_end_index
                target, target_and_negatives = self._prepare_sample(sentence, start, end, homograph, word_id)
                self.data.append({"input": sentence, "target": target, "target_and_negatives": target_and_negatives})
            print("--->", len(self.data), "aligner")


    def _prepare_sample(self, sentence, start, end, homograph, word_id):
        homograph_span = sentence[start:end]
        l_context_len = len(self.tokenizer.tokenize(sentence[:start], add_special_tokens=False))
        r_context_len = len(self.tokenizer.tokenize(sentence[end:], add_special_tokens=False))

        grapheme_ipa_forms = self.wiki_homograph_dict[homograph]
        target_len = len(self.tokenizer.tokenize(homograph_span, add_special_tokens=False))
        target = self.target_ipa.index(grapheme_ipa_forms[word_id])

        # add extra -100 tokens at the begging and end for [CLS] and [SEP] bert tokens, TODO: remove hardcoding
        target = (
                [-100] * (l_context_len + 1) + [target] + [-100] * (target_len - 1) + [-100] * (r_context_len + 1)
        )

        target_and_negatives = [self.target_ipa.index(ipa_) for wordid_, ipa_ in grapheme_ipa_forms.items()]
        return target, target_and_negatives

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
        # Encode inputs (graphemes)
        input_batch = [entry["input"] for entry in batch]
        input_encoding = self.tokenizer(
            input_batch, padding='longest', max_length=self.max_seq_len, truncation=True, return_tensors='pt',
        )

        input_ids, attention_mask = input_encoding.input_ids, input_encoding.attention_mask

        # Encode targets

        targets = [entry["target"] for entry in batch]
        targets_len = [len(entry) for entry in targets]
        max_target_len = max(targets_len)
        targets = [entry + [-100] * (max_target_len - entry_len) for entry, entry_len in zip(targets, targets_len)]
        targets = torch.tensor(targets)

        target_and_negatives = [entry["target_and_negatives"] for entry in batch]

        # create a mask 1 for target_and_negatives and 0 for the rest of the entries since they're not relevant to the target word
        batch_size = input_ids.shape[0]
        num_classes = len(self.target_ipa_label_to_id)
        target_and_negatives_mask = torch.zeros(batch_size, num_classes)
        for i, values in enumerate(target_and_negatives):
            for v in values:
                target_and_negatives_mask[i][v] = 1

        output = (input_ids, attention_mask, target_and_negatives_mask, targets)

        if input_ids.shape != targets.shape:
            import pdb

            pdb.set_trace()
            print()

        return output


class G2PClassificationInferDataset(Dataset):
    """
    Creates dataset to use to train a DuplexDecoderModel.
    Converts from raw data to an instance that can be used by Dataloader.
    For dataset to use to do end-to-end inference, see TextNormalizationTestDataset.

    Args:
        # input_file: path to the raw data file (e.g., train.tsv).
        # 	For more info about the data format, refer to the
        # 	`text_normalization doc <https://github.com/NVIDIA/NeMo/blob/main/docs/source/nlp/text_normalization.rst>`.
        # raw_instances: processed raw instances in the Google TN dataset format (used for tarred dataset)
        # tokenizer: tokenizer of the model that will be trained on the dataset
        # tokenizer_name: name of the tokenizer,
        # mode: should be one of the values ['tn', 'itn', 'joint'].  `tn` mode is for TN only.
        # 	`itn` mode is for ITN only. `joint` is for training a system that can do both TN and ITN at the same time.
        # max_len: maximum length of sequence in tokens. The code will discard any training instance whose input or
        # 	output is longer than the specified max_len.
        # decoder_data_augmentation (bool): a flag indicates whether to augment the dataset with additional data
        # 	instances that may help the decoder become more robust against the tagger's errors.
        # 	Refer to the doc for more info.
        # lang: language of the dataset
        # use_cache: Enables caching to use pickle format to store and read data from
        # max_insts: Maximum number of instances (-1 means no limit)
        # do_tokenize: Tokenize each instance (set to False for Tarred dataset)
        # initial_shuffle: Set to True to shuffle the data
    """

    def __init__(
        self,
        sentences: str,
        start_end: List[int],
        homographs: List[str],
        tokenizer: PreTrainedTokenizerBase,
        wordid_map: str = "wordids.tsv",
        context_len: int = 3,
        max_seq_len: int = 512,
        with_labels: bool = True,
    ):
        """
        # TODO: docstring
        :type input_file: object
        """

        super().__init__()

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.data = []
        self.pad_token = 0
        self.with_labels = with_labels

        wiki_homograph_dict, target_ipa, self.target_ipa_label_to_id = read_wordids(wordid_map)

        num_removed_or_truncated = 0
        if len(sentences) != len(start_end):
            raise ValueError(f"Number of sentences does not match start-end")

        for i, sent in enumerate(sentences):
            start, end = start_end[i]

            # dataset errors fix
            sent, start, end = correct_wikihomograph_data(sent, start, end)
            homograph = homographs[i]
            homograph_span = sent[start:end]
            if homograph_span.lower() != homograph and sent.lower().count(homograph) == 1:
                start = sent.lower().index(homograph)
                end = start + len(homograph)
                homograph_span = sent[start:end].lower()
                assert homograph == homograph_span.lower()

            # we'll skip examples where start/end indices are incorrect and
            # the target homorgaph is also present in the context (ambiguous)
            if homograph_span.lower() == homograph:
                try:
                    grapheme_ipa_forms = wiki_homograph_dict[homograph]
                except:
                    import pdb

                    pdb.set_trace()
                target_and_negatives = [target_ipa.index(ipa_) for wordid_, ipa_ in grapheme_ipa_forms.items()]

                # add 1 to left and right context for [CLS] and [SEP] BERT special tokens added to input
                # during tokenization in _collate_fn()
                l_context_len = len(tokenizer.tokenize(sent[:start], add_special_tokens=False)) + 1
                r_context_len = len(tokenizer.tokenize(sent[end:], add_special_tokens=False)) + 1
                homograph_len = len(tokenizer.tokenize(homograph_span, add_special_tokens=False))
                # use prediction of the first subword of the homograph
                homograph_mask = [1] + [0] * (homograph_len - 1)
                pred_mask = [0] * l_context_len + homograph_mask + [0] * r_context_len
                self.data.append({"input": sent, "target_and_negatives": target_and_negatives, "pred_mask": pred_mask})
            else:
                import pdb

                pdb.set_trace()
                num_removed_or_truncated += 1

        logging.info(f"Number of samples removed or truncated {num_removed_or_truncated} examples")

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
        # Encode inputs (graphemes)
        input_batch = [entry["input"] for entry in batch]
        input_encoding = self.tokenizer(
            input_batch, padding='longest', max_length=self.max_seq_len, truncation=True, return_tensors='pt',
        )
        input_ids, attention_mask = input_encoding.input_ids, input_encoding.attention_mask

        # Encode targets
        target_and_negatives = [entry["target_and_negatives"] for entry in batch]

        # create a mask 1 for target_and_negatives and 0 for the rest of the entries since they're not relevant to the target word
        batch_size = input_ids.shape[0]
        num_classes = len(self.target_ipa_label_to_id)
        target_and_negatives_mask = torch.zeros(batch_size, num_classes)
        for i, values in enumerate(target_and_negatives):
            for v in values:
                target_and_negatives_mask[i][v] = 1

        # pad prediction mask (masks out irrelevant words)
        pred_mask = [entry["pred_mask"] for entry in batch]
        pred_mask_len = [len(entry) for entry in pred_mask]
        max_pred_mask_len = max(pred_mask_len)
        pred_mask = [
            entry + [0] * (max_pred_mask_len - entry_len) for entry, entry_len in zip(pred_mask, pred_mask_len)
        ]
        pred_mask = torch.tensor(pred_mask)

        output = (input_ids, attention_mask, target_and_negatives_mask, pred_mask)
        return output
