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

from nemo.core.classes import Dataset
from nemo.utils import logging

__all__ = ['correct_wikihomograph_data', 'read_wikihomograph_file', 'read_wordids']


def correct_wikihomograph_data(sentence, start=None, end=None):
    """
    Correct indices for wikihomograph data
    :param sentence:
    :param start:
    :param end:
    :return:
    """
    corrections = {
        "It is traditionally composed of 85–99% tin, mixed with copper, antimony, bismuth, and sometimes lead, although the use of lead is less common today.": [
            96,
            100,
        ],
        "B₁₀₅ can be conceptually divided into a B₄₈ fragment and B₂₈-B-B₂₈ (B₅₇) fragment.": [44, 52],
        "Pierrefonds Airport on Réunion recorded just 18 mm (0.71 in) of rainfall from November to January, a record minimum.": [
            101,
            107,
        ],
        "Consort Chen Farong (陳法容) was an imperial consort during the Chinese dynasty Liu Song.": [42, 49],
        "Unlike TiO₂, which features six-coordinate Ti in all phases, monoclinic zirconia consists of seven-coordinate zirconium centres.": [
            32,
            42,
        ],
        "Its area is 16 km², its approximate length is 10 km, and its approximate width is 3 km.": [24, 35],
        "The conjugate momentum to X has the expressionwhere the pᵢ are the momentum functions conjugate to the coordinates.": [
            86,
            95,
        ],
        "Furthermore 17β-HSD1 levels positively correlate with E2 and negatively correlate with DHT levels in breast cancer cells.": [
            39,
            48,
        ],
        "Electric car buyers get a €4,000 (US$4,520) discount while buyers of plug-in hybrid vehicles get a discount of €3,000 (US$3,390).": [
            99,
            107,
        ],
    }

    if sentence in corrections:
        start, end = corrections[sentence]

    sentence = sentence.replace("2014Coordinate", "2014 Coordinate")
    sentence = sentence.replace("AAA", "triple A")

    return sentence, start, end


def read_wikihomograph_file(file: str) -> (List[str], List[List[int]], List[str], List[str]):
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
    excluded_sentences = 0
    num_corrected = 0
    sentences = []
    start_end_indices = []
    homographs = []
    word_ids = []
    with open(file, "r", encoding="utf-8") as f:
        tsv_file = csv.reader(f, delimiter="\t")
        for i, line in enumerate(tsv_file):
            if i == 0:
                continue
            homograph, wordid, sentence, start, end = line
            start, end = int(start), int(end)

            sentence, start, end = correct_wikihomograph_data(sentence, start, end)

            homograph_span = sentence[start:end]
            if homograph_span.lower() != homograph:
                if sentence.lower().count(homograph) == 1:
                    start = sentence.lower().index(homograph)
                    end = start + len(homograph)
                    homograph_span = sentence[start:end].lower()

                    assert homograph == homograph_span.lower()
                else:
                    print(f"homograph {homograph} != homograph_span {homograph_span} in {sentence}")
                    excluded_sentences += 1
                    # # excluded_sentences.append(sentence)
                    # continue
                    import pdb

                    pdb.set_trace()
                    raise ValueError(f"homograph {homograph} != homograph_span {homograph_span} in {sentence}")

            homographs.append(homograph)
            start_end_indices.append([start, end])
            sentences.append(sentence)
            word_ids.append(wordid)

    if num_corrected > 0:
        print(f"corrected: {num_corrected}")
    if excluded_sentences > 0:
        print(f"excluded: {excluded_sentences}")
    return sentences, start_end_indices, homographs, word_ids


def read_wordids(wordid_map):
    wiki_homograph_dict = {}
    target_ipa = []
    target_ipa_label_to_id = {}

    with open(wordid_map, "r", encoding="utf-8") as f:
        tsv_file = csv.reader(f, delimiter="\t")

        for i, line in enumerate(tsv_file):
            if i == 0:
                continue

            grapheme = line[0]
            word_id = line[1]
            ipa_form = line[3]
            target_ipa_label_to_id[word_id] = len(target_ipa)
            target_ipa.append(ipa_form)
            if grapheme not in wiki_homograph_dict:
                wiki_homograph_dict[grapheme] = {}
            wiki_homograph_dict[grapheme][word_id] = ipa_form
    return wiki_homograph_dict, target_ipa, target_ipa_label_to_id
