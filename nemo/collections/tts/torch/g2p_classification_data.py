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


import os
from collections import defaultdict
from glob import glob
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

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
        # IDs for special tokens for encoding inputs of the decoder models
        EXTRA_ID_0 = '<extra_id_0>'
        EXTRA_ID_1 = '<extra_id_1>'

        target_ipa = []
        self.target_ipa_label_to_id = {}
        #  read wordids.tsv file
        wiki_homograph_dict = defaultdict(dict)
        with open(wordid_map, "r") as f_in:
            for i, line in enumerate(f_in):
                if i == 0:
                    continue
                line = line.replace('"', "").strip().split("\t")
                grapheme = line[0].strip()
                word_id = line[1].strip()
                ipa_form = line[3].strip()
                self.target_ipa_label_to_id[word_id] = len(target_ipa)
                target_ipa.append(ipa_form)
                wiki_homograph_dict[grapheme][word_id] = ipa_form
        num_removed_or_truncated = 0

        for file in glob(f"{dir_name}/*.tsv"):
            with open(file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i == 0:
                        continue
                    homograph, wordid, sentence, start, end = line.replace('"', "").strip().split("\t")
                    start, end = int(start), int(end)
                    homograph_span = sentence[start:end].lower()
                    if homograph_span != homograph and sentence.lower().count(homograph) == 1:
                        start = sentence.lower().index(homograph)
                        end = start + len(homograph)
                        homograph_span = sentence[start:end].lower()
                        assert homograph == homograph_span.lower()

                    # we'll skip examples where start/end indices are incorrect and
                    # the target homorgaph is also present in the context (ambiguous)
                    if homograph_span == homograph:
                        try:
                            l_context = sentence[:start].split()[-context_len:]
                        except:
                            import pdb

                            pdb.set_trace()
                        r_context = sentence[end:].split()[:context_len]
                        input = l_context + [EXTRA_ID_0, homograph, EXTRA_ID_1] + r_context
                        input = " ".join(input)
                        # input_sequence = f"sentence1: {input} ipa: {wiki_homograph_dict[homograph][wordid]}"
                        # if target_len > len(item["text_graphemes"]) or target_len > max_target_len:
                        # # num_removed_or_truncated += 1
                        # #                 continue
                        grapheme_ipa_forms = wiki_homograph_dict[homograph]
                        target = target_ipa.index(grapheme_ipa_forms[wordid])
                        target_and_negatives = [target_ipa.index(ipa_) for wordid_, ipa_ in grapheme_ipa_forms.items()]
                        self.data.append({"input": input, "target": target, "target_and_negatives": target_and_negatives})
                        # for wordid_, ipa in wiki_homograph_dict[homograph].items():
                        #     if wordid_ != wordid:
                        #         input_sequence = f"sentence1: {input} ipa: {ipa}"
                        #         self.data.append(
                        #             {"input": input_sequence, "target": 0,}
                        #         )
                    else:
                        num_removed_or_truncated += 1
                        # self.data.append(
                        #                     {
                        #                         "ho": item["text_graphemes"],
                        #                         "phonemes": item["text"],
                        #                         "target": target_tokens,
                        #                         "target_len": target_len,
                        #                     })

        # import pdb;
        # pdb.set_trace()
        logging.info(f"Number of samples removed or truncated {num_removed_or_truncated} examples from {dir_name}")

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
        # import pdb; pdb.set_trace()
        # Encode inputs (graphemes)
        input_batch = [entry["input"] for entry in batch]
        input_encoding = self.tokenizer(
            input_batch, padding='longest', max_length=self.max_seq_len, truncation=True, return_tensors='pt',
        )
        input_ids, attention_mask = input_encoding.input_ids, input_encoding.attention_mask

        # Encode targets
        targets = torch.tensor([entry["target"] for entry in batch])
        target_and_negatives = torch.tensor([entry["target_and_negatives"] for entry in batch])
        # import pdb;
        # pdb.set_trace()
        output = (input_ids, attention_mask, targets)

        # graphemes_batch = [entry["graphemes"] for entry in batch]
        #
        # # Encode inputs (graphemes)
        # if isinstance(self.tokenizer_graphemes, PreTrainedTokenizerBase):
        #     input_encoding = self.tokenizer_graphemes(
        #         graphemes_batch,
        #         padding='longest',
        #         max_length=self.max_source_len,
        #         truncation=True,
        #         return_tensors='pt',
        #     )
        #     input_ids, attention_mask = input_encoding.input_ids, input_encoding.attention_mask
        #     input_len = torch.sum(attention_mask, 1) - 1
        # else:
        #     input_ids = [self.tokenizer_graphemes.text_to_ids(sentence) for sentence in graphemes_batch]
        #     input_len = [len(entry) for entry in input_ids]
        #     max_len = max(input_len)
        #     input_ids = [entry + [0] * (max_len - entry_len) for entry, entry_len in zip(input_ids, input_len)]
        #     attention_mask = None  # not used with Conformer encoder
        #     input_ids = torch.tensor(input_ids)
        #     input_len = torch.tensor(input_len)
        #
        # # inference
        # if not self.with_labels:
        #     output = (input_ids, attention_mask, input_len)
        # # Encode targets (phonemes)
        # else:
        #     targets = [torch.tensor(entry["target"]) for entry in batch]
        #     target_lengths = [torch.tensor(entry["target_len"]) for entry in batch]
        #     max_target_len = max(target_lengths)
        #
        #     padded_targets = []
        #     for target, target_len in zip(targets, target_lengths):
        #         pad = (0, max_target_len - target_len)
        #         target_pad = torch.nn.functional.pad(target, pad, value=len(self.labels))
        #         padded_targets.append(target_pad)
        #
        #     padded_targets = torch.stack(padded_targets)
        #     target_lengths = torch.stack(target_lengths)
        #     output = (input_ids, attention_mask, input_len, padded_targets, target_lengths)

        return output


if __name__ == "__main__":
    base_dir = "/Users/ebakhturina/g2p_scripts/WikipediaHomographData-master/data/"
    wordid_map = f"{base_dir}/wordids.tsv"
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    dataset = G2PClassificationDataset(dir_name=f"{base_dir}/DEL", wordid_map=wordid_map, tokenizer=tokenizer)

    import pdb

    pdb.set_trace()
    print()
