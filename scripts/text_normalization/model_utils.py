# Copyright 2022, NVIDIA CORPORATION & AFFILIATION.
# Copyright 2020 AWSLABS, AMAZON.
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
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForMaskedLM

from torch.nn.functional import softmax
import torch
from nemo.utils import logging
import math
from typing import List
import re
from tqdm import tqdm
import numpy as np




class _MLMScorer:
    def __init__(self, model_name, device='cpu'):
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.device = device
        self.MASK_LABEL = self.tokenizer.mask_token

    def score_sentences(self, sentences: List[str]):
        """
        TODO: Optimize for duplicates (in list)
        """

        return [self.score_sentence(sentence) for sentence in sentences]

    def score_sentence(self, sentence: str):
        """
        TODO: Optimize for duplicates (after mapping)
        """
        assert type(sentence) == str

        tokens = self.tokenizer.tokenize(sentence)
        mask_idx = []
        token_type = []
        attn_mask = []
        ids = []
        for m_idx, _ in enumerate(tokens):
            masked = self.__mask_text__(m_idx, tokens)
            mask_idx.append(m_idx)
            ids.append(self.tokenizer.encode(masked))
            id_len = len(ids[-1])
            token_type.append([0] * id_len)
            attn_mask.append([1] * id_len)


        data = {
            'input_ids': torch.tensor(ids, device=self.device),
            'attention_mask': torch.tensor(attn_mask, device=self.device),
            'token_type_ids': torch.tensor(token_type, device=self.device),
        }

        with torch.no_grad():
            outputs = self.model(**data)
            logits = outputs.logits

        scores = []
        scores_log_prob = 0.0

        for i, m_idx in enumerate(mask_idx):
            preds = logits[i].squeeze(0)
            probs = softmax(preds, dim=1)
            token_id = self.tokenizer.convert_tokens_to_ids([tokens[m_idx]])[0]
            log_prob = np.log(probs[m_idx + 1, token_id].cpu().numpy()).item()
            scores.append(log_prob)
            scores_log_prob += log_prob

        return scores_log_prob


    def __mask_text__(self, idx, tokens):
        masked = tokens.copy()
        masked[idx] = self.MASK_LABEL
        return masked




def init_models(model_name_list):
    model_names = model_name_list.split(",")
    models ={}
    for model_name in model_names:
        models[model_name] = _MLMScorer(model_name)
    return models

def get_model_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelWithLMHead.from_pretrained(model_name)
    print(
        f"MODEL: {model_name}, NUM_PARAMETERS: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return (model, tokenizer)


def get_score(text, model, do_lower=True):
    try:
        if isinstance(text, str):
            text = [text.lower() if do_lower else text]
        else:
            text = [t.lower() for t in text] if do_lower else text
        score = -1 * sum(model.score_sentences(text)) / len(text)
    except Exception as e:
        print(e)
        print(f"Scoring error: {text}")
        score = math.inf
    return score

def get_masked_score(text, model, do_lower=True):
    spans = re.findall("<\s[^>.]*\s>", text)
    if len(spans) > 0:
        text_with_mask = []
        for span in spans:
            text_with_mask.append(text.replace(span, model.MASK_LABEL).replace("< ", "").replace(" >", ""))
        text = text_with_mask
    return get_score(text, model, do_lower=do_lower)


def score_options(sentences: List[str], context_len, model, do_lower=True):
    scores = []
    if context_len is not None:
        diffs = [find_diff(s, context_len) for s in sentences]
        if len(set([len(d) for d in diffs])) == 1:
            sentences = diffs
    for sent in tqdm(sentences):
        bos = ""
        if isinstance(sent, list): # in case of set context len
            option_scores = [get_masked_score(bos + s, model, do_lower) for s in sent]
            logging.debug(sent)
            logging.debug(option_scores)
            logging.debug("=" * 50)
            if any(math.isnan(x) for x in option_scores):
                av_score = math.inf
            else:
                av_score = round(sum(option_scores) / len(option_scores), 4)
            scores.append(av_score)
        elif isinstance(sent, str): # in case of full context
            scores.append(round(get_masked_score(bos + sent, model, do_lower=do_lower)))
        else:
            raise ValueError()
    return scores


def find_diff(text, context_len=3):
    """Finds parts of text normalized by WFST and returns them in list with a context of context_len"""
    diffs = []
    pattern_start = "< "
    pattern_end = " >"
    def __clean(s):
        return s.replace(pattern_start, "").replace(pattern_end, "").replace("  ", " ")

    index_start = 0
    while pattern_start in text[index_start:]:
        index_start = index_start + text[index_start:].index(pattern_start)
        offset = index_start
        if pattern_end in text[offset:]:
            index_end = offset + text[offset:].index(pattern_end) + len(pattern_end)
            center = __clean(text[index_start: index_end])

            left_context = " ".join(__clean(text[:index_start]).split()[-context_len:])
            if len(left_context) > 0 and text[:index_start][-1].isspace():
                left_context = left_context + " "
            right_context = " ".join(__clean(text[index_end:]).split()[:context_len])
            if len(right_context) > 0 and text[index_end][0].isspace():
                right_context = " " + right_context
            diffs.append(left_context + center + right_context)
            index_end += 1
            index_start = index_end + 1
        else:
            break
    if len(diffs) == 0:
        diffs = [text]
    return diffs