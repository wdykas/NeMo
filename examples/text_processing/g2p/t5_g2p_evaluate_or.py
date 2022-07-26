# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
"""
Evaluation script for T5 G2P checkpoints.
"""

import json
import sys
from argparse import ArgumentParser

import torch
from examples.tts.G2P.heteronyms_correction_with_classification import clean, correct_heteronyms, get_metrics
from nemo_text_processing.g2p.models.t5_g2p import T5G2PModel
from omegaconf import OmegaConf

sys.path.append("/home/ebakhturina/NeMo/examples/tts/G2P")


try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


can_gpu = torch.cuda.is_available()

parser = ArgumentParser()
parser.add_argument(
    "--model_ckpt", type=str, required=True, help="T5 G2P model checkpoint for prediction",
)
parser.add_argument("--manifest_filepath", type=str, required=True, help="Path to evaluation data")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument(
    "--output", type=str, default="T5_generative_predictions.json", help="Path to store model's predictions"
)
parser.add_argument(
    "--clean",
    action="store_true",
    help="Set to True to lower case input graphemes and remove punctuation marks for evaluation",
)
parser.add_argument("--per_word", action="store_true", help="Set to True to run inference per word")
parser.add_argument(
    "--heteronyms_model",
    default=None,
    type=str,
    help="Path to a pre-trained classification model for heteronyms disambiguation",
)
parser.add_argument(
    "--grapheme_field",
    default="text_graphemes",
    type=str,
    help="Name of the field in the manifest to load grapheme input from ",
)


def main():
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    # Restore model
    g2p_model = T5G2PModel.restore_from(restore_path=args.model_ckpt)

    # Prepare test data
    def _create_test(manifest, batch_size, num_workers):
        test_config = OmegaConf.create(
            {
                'dataset': None,
                'manifest_filepath': manifest,
                'dataloader_params': {
                    'drop_last': False,
                    'shuffle': False,
                    'batch_size': batch_size,
                    'num_workers': num_workers,
                },
            }
        )
        return test_config

    if can_gpu:
        g2p_model = g2p_model.cuda()
    g2p_model.eval()

    g2p_model.max_source_len = 512
    if args.per_word:
        with open(args.output, "w", encoding="utf-8") as f_out, open(args.manifest_filepath, "r") as f_in:
            for line in f_in:
                line = json.loads(line)

                clean_graphemes = clean(line["text_graphemes"])
                clean_phonemes = clean(line["text"])
                line["clean_graphemes"] = clean_graphemes
                line["original_text"] = line["text"]
                line["text"] = clean_phonemes

                tmp_file = "/tmp/tmp.json"
                with open(tmp_file, "w") as f_tmp:
                    clean_graphemes = clean_graphemes.split()
                    for g in clean_graphemes:
                        entry = {"text_graphemes": g, "text": "n/a"}
                        f_tmp.write(json.dumps(entry, ensure_ascii=False) + "\n")

                tmp_test_config = _create_test(tmp_file, args.batch_size, args.num_workers)
                g2p_model.setup_test_data(tmp_test_config)
                hypotheses = []
                for test_batch in g2p_model.test_dataloader():
                    if can_gpu:
                        test_batch = [x.cuda() for x in test_batch]
                    with autocast():
                        input_ids, _, _ = test_batch
                        # Get predicted phonemes
                        generated_str, _, _ = g2p_model._generate_predictions(
                            input_ids=input_ids, model_max_target_len=g2p_model.max_target_len
                        )
                        hypotheses += generated_str

                preds = " ".join(hypotheses)
                line["pred_text"] = preds
                f_out.write(json.dumps(line, ensure_ascii=False) + "\n")
        get_metrics(args.output)
    else:
        test_config = _create_test(args.manifest_filepath, args.batch_size, args.num_workers)
        g2p_model.setup_test_data(test_config)

        # Get predictions
        orig_sentences = []
        hypotheses = []
        references = []
        for test_batch in g2p_model.test_dataloader():
            if can_gpu:
                test_batch = [x.cuda() for x in test_batch]
            with autocast():
                input_ids, _, labels = test_batch
                # Get reference phonemes
                labels_str = g2p_model._tokenizer.batch_decode(
                    # Need to do the following to zero out the -100s (ignore_index).
                    torch.ones_like(labels) * ((labels == -100) * 100) + labels,
                    skip_special_tokens=True,
                )
                references += labels_str

                # Get predicted phonemes
                generated_str, _, _ = g2p_model._generate_predictions(
                    input_ids=input_ids, model_max_target_len=g2p_model.max_target_len
                )
                hypotheses += generated_str

                # Get original graphemes
                graphemes = g2p_model._tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                orig_sentences += graphemes

        with open(args.output, "w", encoding="utf-8") as f_out, open(args.manifest_filepath, "r") as f_in:
            for i, line in enumerate(f_in):
                line = json.loads(line)
                line["pred_text"] = hypotheses[i]
                line["duration"] = 0.1
                if args.clean:
                    line["pred_text"] = clean(line["pred_text"])
                    line["text"] = clean(line["text"])
                f_out.write(json.dumps(line, ensure_ascii=False) + "\n")

        get_metrics(args.output)
        print(f"Predictions saved in {args.output}")
        # # Print eval results
        # per = word_error_rate(hypotheses=hypotheses, references=references)
        # print(f"PER: {per}")
        #
        # for h, r, s in zip(hypotheses, references, orig_sentences):
        #     if h != r:
        #         print("----")
        #         print(s)
        #         print(h)
        #         print(r)

        del g2p_model
        # if False:
        #     pass
        # else:

        args.output = "/home/ebakhturina/CharsiuG2P/path_to_output/eval_wikihomograph.json_byt5-small.json"
        if args.heteronyms_model is not None:
            correct_heteronyms(args.heteronyms_model, args.output, args.batch_size)


if __name__ == '__main__':
    main()
