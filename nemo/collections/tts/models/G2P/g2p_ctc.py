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

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from torch import nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models import EncDecCTCModel
from nemo.collections.asr.models.asr_model import ASRModel, ExportableEncDecModel
from nemo.collections.asr.parts.mixins import ASRBPEMixin
from nemo.collections.common.tokenizers.char_tokenizer import CharTokenizer
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.tts.torch.data import CTCG2PBPEDataset
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.core.neural_types import LabelsType, LossType, MaskType, NeuralType, TokenIndex
from nemo.utils import logging

__all__ = ['CTCG2PModel']


@dataclass
class CTCG2PConfig:
    train_ds: Optional[Dict[Any, Any]] = None
    validation_ds: Optional[Dict[Any, Any]] = None


class CTCG2PModel(ModelPT, ASRBPEMixin):  # ! ASR dependency here
    """
    CTC-based grapheme-to-phoneme model.
    """

    # @property
    # def input_types(self) -> Optional[Dict[str, NeuralType]]:
    #     return {
    #         "input_ids": NeuralType(('B', 'T'), TokenIndex()),
    #         "attention_mask": NeuralType(('B', 'T'), MaskType(), optional=True),
    #         "labels": NeuralType(('B', 'T'), LabelsType()),
    #     }
    #
    # @property
    # def output_types(self) -> Optional[Dict[str, NeuralType]]:
    #     return {"loss": NeuralType((), LossType())}

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_gpus

        self.mode = cfg.model_name.lower()
        supported_modes = ["byt5", "conformer_bpe"]
        if self.mode not in supported_modes:
            raise ValueError(f"{self.mode} is not supported, choose from {supported_modes}")

        if self.mode == "byt5":
            # Load appropriate tokenizer from HuggingFace
            self.tokenizer_grapheme = AutoTokenizer.from_pretrained(cfg.tokenizer_grapheme.pretrained)
            self.max_source_len = cfg.get("max_source_len", self.tokenizer_grapheme.model_max_length)
            self.max_target_len = cfg.get("max_target_len", self.tokenizer_grapheme.model_max_length)
        elif "conformer" in self.mode:
            self.max_source_len = cfg.get("max_source_len", 512)
            self.max_target_len = cfg.get("max_target_len", 512)
            # set up grapheme tokenizer
            chars = [
                " ",
                "a",
                "b",
                "c",
                "d",
                "e",
                "f",
                "g",
                "h",
                "i",
                "j",
                "k",
                "l",
                "m",
                "n",
                "o",
                "p",
                "q",
                "r",
                "s",
                "t",
                "u",
                "v",
                "w",
                "x",
                "y",
                "z",
                "'",
            ]
            vocab_file = "/tmp/char_vocab.txt"
            with open(vocab_file, "w") as f:
                [f.write(f'"{ch}"\n') for ch in chars]

            self.tokenizer_grapheme = CharTokenizer(vocab_file=vocab_file)

        # Setup phoneme tokenizer
        self._setup_tokenizer(cfg.tokenizer)

        # Initialize vocabulary
        vocabulary = self.tokenizer.tokenizer.get_vocab()
        cfg.decoder.vocabulary = ListConfig(list(vocabulary.keys()))
        self.vocabulary = cfg.decoder.vocabulary
        self.labels_tkn2id = {l: i for i, l in enumerate(self.vocabulary)}
        self.labels_id2tkn = {i: l for i, l in enumerate(self.vocabulary)}

        super().__init__(cfg, trainer)

        self._setup_encoder()
        self.decoder = EncDecCTCModel.from_config_dict(self._cfg.decoder)
        self.loss = CTCLoss(
            num_classes=self.decoder.num_classes_with_blank - 1,
            zero_infinity=True,
            reduction=self._cfg.get("ctc_reduction", "mean_batch"),
        )
        logging.info("\n\n----> DONE with init()\n\n")

    def _setup_encoder(self):
        if self.mode == "byt5":
            config = AutoConfig.from_pretrained(self._cfg.tokenizer_grapheme.pretrained)
            if self._cfg.encoder.dropout is not None:
                config.dropout_rate = self._cfg.encoder.dropout
                print(f"\nDROPOUT: {config.dropout_rate}")
            self.encoder = AutoModel.from_pretrained(self._cfg.encoder.transformer, config=config).encoder
            # add encoder hidden dim size to the config
            if self.cfg.decoder.feat_in is None:
                self._cfg.decoder.feat_in = self.encoder.config.d_model
        else:
            self.embedding = nn.Embedding(
                embedding_dim=self._cfg.embedding.d_model, num_embeddings=self.tokenizer.vocab_size, padding_idx=0
            )
            self.encoder = EncDecCTCModel.from_config_dict(self._cfg.encoder)
            with open_dict(self._cfg):
                if "feat_in" not in self._cfg.decoder or (
                    not self._cfg.decoder.feat_in and hasattr(self.encoder, '_feat_out')
                ):
                    self._cfg.decoder.feat_in = self.encoder._feat_out
                if "feat_in" not in self._cfg.decoder or not self._cfg.decoder.feat_in:
                    raise ValueError("param feat_in of the decoder's config is not set!")

    # @typecheck()
    def forward(self, input_ids, attention_mask, input_len):
        if self.mode == "byt5":
            encoded_input = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
            encoded_len = input_len
            # encoded_input = [B, seq_len, hid_dim]
            # swap seq_len and hid_dim dimensions to get [B, hid_dim, seq_len]
            encoded_input = encoded_input.transpose(1, 2)
        else:
            input_embedding = self.embedding(input_ids)
            input_embedding = input_embedding.transpose(1, 2)
            encoded_input, encoded_len = self.encoder(audio_signal=input_embedding, length=input_len)

        log_probs = self.decoder(encoder_output=encoded_input)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)
        return log_probs, greedy_predictions, encoded_len

    # ===== Training Functions ===== #
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, input_len, targets, target_lengths = batch

        log_probs, predictions, encoded_len = self.forward(
            input_ids=input_ids, attention_mask=attention_mask, input_len=input_len
        )

        loss = self.loss(
            log_probs=log_probs, targets=targets, input_lengths=encoded_len, target_lengths=target_lengths
        )
        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outputs):
        return super().training_epoch_end(outputs)

    # ===== Validation Functions ===== #
    def validation_step(self, batch, batch_idx, split="val"):
        input_ids, attention_mask, input_len, targets, target_lengths = batch

        log_probs, greedy_predictions, encoded_len = self.forward(
            input_ids=input_ids, attention_mask=attention_mask, input_len=input_len
        )
        val_loss = self.loss(
            log_probs=log_probs, targets=targets, input_lengths=encoded_len, target_lengths=target_lengths
        )

        preds_str = self.ctc_decoder_predictions_tensor(greedy_predictions.tolist())
        targets_str = [self.decode_ids_to_str(t) for t in targets.tolist()]
        per = word_error_rate(hypotheses=preds_str, references=targets_str)
        self.log(f"{split}_loss", val_loss)
        return {f"{split}_loss": val_loss, "per": per}

    def test_step(self, batch, batch_idx):
        """
        Lightning calls this inside the test loop with the data from the test dataloader
        passed in as `batch`.
        """
        return self.validation_step(batch, batch_idx, split="test")

    def decode_ids_to_str(self, ids: List[int]) -> str:
        blank_id = len(self.labels_id2tkn)
        ids = [t for t in ids if t != blank_id]
        decoded = self.tokenizer.ids_to_text(ids)
        return decoded

    def ctc_decoder_predictions_tensor(self, predictions: List[List[int]], predictions_len=None) -> List[str]:
        """
        Decodes a sequence of labels to words

        Args:
            predictions: An integer torch.Tensor of shape [Batch, Time] (if ``batch_index_dim == 0``) or [Time, Batch]
                (if ``batch_index_dim == 1``) of integer indices that correspond to the index of some character in the
                label set.
            predictions_len: Optional tensor of length `Batch` which contains the integer lengths
                of the sequence in the padded `predictions` tensor.
            return_hypotheses: Bool flag whether to return just the decoding predictions of the model
                or a Hypothesis object that holds information such as the decoded `text`,
                the `alignment` of emited by the CTC Model, and the `length` of the sequence (if available).
                May also contain the log-probabilities of the decoder (if this method is called via
                transcribe())

        Returns:
            Either a list of str which represent the CTC decoded strings per sample,
            or a list of Hypothesis objects containing additional information.
        """

        blank_id = len(self.labels_id2tkn)
        hypotheses = []

        # iterate over batch
        for ind, prediction in enumerate(predictions):
            if predictions_len is not None:
                prediction = prediction[: predictions_len[ind]]

            # CTC decoding procedure
            decoded_prediction = []
            previous = blank_id
            for p in prediction:
                if (p != previous or previous == blank_id) and p != blank_id:
                    decoded_prediction.append(p)
                previous = p

            text = self.decode_ids_to_str(decoded_prediction)
            hypotheses.append(text)
        return hypotheses

    def validation_epoch_end(self, outputs, split="val"):
        """
        Called at the end of validation to aggregate outputs (reduces across batches, not workers).
        """
        # TODO: Add support for multi GPU
        avg_loss = torch.stack([x[f"{split}_loss"] for x in outputs]).mean()
        self.log(f"{split}_loss", avg_loss, prog_bar=True)

        # TODO: Add better PER calculation and logging.
        avg_per = sum([x["per"] for x in outputs]) / len(outputs)
        self.log(f"{split}_per", avg_per)
        logging.info(f"---------------> PER: {round(avg_per*100, 2)}%")

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs, split="test")

    def _setup_infer_dataloader(
        self, manifest_filepath: str, batch_size: int, num_workers: int
    ) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a infer data loader.

        Args:
            queries: text
            batch_size: batch size to use during inference

        Returns:
            A pytorch DataLoader.
        """
        dataset = CTCG2PBPEDataset(
            manifest_filepath=manifest_filepath,
            tokenizer_graphemes=self.tokenizer_grapheme,
            tokenizer_phonemes=self.tokenizer,
            labels=self.vocabulary,
            max_source_len=self._cfg.max_source_len,
            with_labels=False,
        )

        return torch.utils.data.DataLoader(
            dataset,
            collate_fn=dataset.collate_fn,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

    @torch.no_grad()
    def _infer(self, manifest_filepath: str, batch_size: int, num_workers: int = 0) -> List[int]:
        """
        Get prediction for the queries
        Args:
            queries: text sequences
            batch_size: batch size to use during inference.
        Returns:
        all_preds: model predictions
        """
        # store predictions for all queries in a single list
        all_preds = []
        mode = self.training
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # Switch model to evaluation mode
            self.eval()
            self.to(device)
            infer_datalayer = self._setup_infer_dataloader(
                manifest_filepath, batch_size=batch_size, num_workers=num_workers
            )

            for batch in tqdm(infer_datalayer):
                input_ids, attention_mask, input_len = batch

                log_probs, greedy_predictions, encoded_len = self.forward(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    input_len=input_len.to(device),
                )
                preds_str = self.ctc_decoder_predictions_tensor(greedy_predictions.tolist())
                all_preds.extend(preds_str)

                del greedy_predictions
                del log_probs
                del batch
                del input_len
        finally:
            # set mode back to its original value
            self.train(mode=mode)
        return all_preds

    # Functions for inference
    @torch.no_grad()
    def convert_graphemes_to_phonemes(
        self,
        manifest_filepath: str,
        batch_size: int = 32,
        num_workers: int = 0,
        output_file: Optional[str] = None,
        target_field: Optional[str] = None,
    ) -> List[str]:
        """ Main function for Inference
        Args:
            sents: A list of inputs tokenized by a basic tokenizer.
            nb_spans: A list of ints where each int indicates the number of semiotic spans in each input.
            span_starts: A list of lists where each list contains the starting locations of semiotic spans in an input.
            span_ends: A list of lists where each list contains the ending locations of semiotic spans in an input.
            inst_directions: A list of str where each str indicates the direction of the corresponding instance (i.e., INST_BACKWARD for ITN or INST_FORWARD for TN).

        Returns: A list of lists where each list contains the decoded spans for the corresponding input.
        """
        if not os.path.exists(manifest_filepath):
            raise ValueError(f"{manifest_filepath} is not found")

        all_preds = self._infer(manifest_filepath, batch_size=batch_size, num_workers=num_workers)

        if output_file is None:
            output_file = manifest_filepath.replace(".json", "_phonemes.json")

        logging.info(f"Saving predictions to {output_file}.")
        all_targets = []
        with open(manifest_filepath, "r") as f_in:
            with open(output_file, 'w', encoding="utf-8") as f_out:
                for i, line in tqdm(enumerate(f_in)):
                    line = json.loads(line)

                    if target_field is not None:
                        if target_field not in line:
                            if i == 0:
                                logging.error(
                                    f"{target_field} not found in {manifest_filepath}. Skipping PER calculation"
                                )
                        else:
                            line["graphemes"] = line["text"]
                            line["text"] = line[target_field]
                            line["PER"] = word_error_rate(hypotheses=[all_preds[i]], references=[line[target_field]])
                            all_targets.append(line[target_field])

                    line["pred_text"] = all_preds[i]
                    f_out.write(json.dumps(line) + "\n")

        if target_field is not None:
            per = word_error_rate(hypotheses=all_preds, references=all_targets)
            logging.info(f"Overall PER --- {round(per * 100, 2)}%")
        logging.info(f"Predictions saved to {output_file}.")
        return all_preds

    # ===== Dataset Setup Functions ===== #
    def _setup_dataloader_from_config(self, cfg, name):
        if "dataset" not in cfg or not isinstance(cfg.dataset, DictConfig):
            raise ValueError(f"No dataset for {name}")
        if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
            raise ValueError(f"No dataloader_params for {name}")

        dataset = CTCG2PBPEDataset(
            manifest_filepath=cfg.dataset.manifest_filepath,
            tokenizer_graphemes=self.tokenizer_grapheme,
            tokenizer_phonemes=self.tokenizer,
            labels=self.vocabulary,
            max_source_len=self.max_source_len,
            max_target_len=self.max_target_len,
            with_labels=True,
        )

        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)

    def setup_training_data(self, cfg):
        if not cfg or cfg.dataset.manifest_filepath is None or not os.path.exists(cfg.dataset.manifest_filepath):
            logging.info(
                f"Dataloader config or file_path for the train is missing, so no data loader for train is created!"
            )
            self._train_dl = None
            return
        self._train_dl = self._setup_dataloader_from_config(cfg, name="train")

    def setup_validation_data(self, cfg):
        if not cfg or cfg.dataset.manifest_filepath is None or not os.path.exists(cfg.dataset.manifest_filepath):
            logging.info(
                f"Dataloader config or file_path for the validation is missing, so no data loader for validation is created!"
            )
            self._validation_dl = None
            return
        self._validation_dl = self._setup_dataloader_from_config(cfg, name="validation")

    def setup_test_data(self, cfg):
        if not cfg or cfg.dataset.manifest_filepath is None or not os.path.exists(cfg.dataset.manifest_filepath):
            logging.info(
                f"Dataloader config or file_path for the test is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._test_dl = self._setup_dataloader_from_config(cfg, name="test")

    # ===== List Available Models - N/A =====$
    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        return []
