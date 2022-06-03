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
from time import perf_counter
from typing import Dict, List, Optional

import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch import nn
from transformers import AutoModel, AutoTokenizer, DataCollatorForTokenClassification
from transformers.tokenization_utils_base import BatchEncoding

from nemo.collections.common.losses import CrossEntropyLoss
from nemo.collections.nlp.data.text_classification import TextClassificationDataset, calc_class_weights
from nemo.collections.nlp.metrics.classification_report import ClassificationReport
from nemo.collections.nlp.models.duplex_text_normalization.utils import has_numbers
from nemo.collections.nlp.modules.common import SequenceClassifier
from nemo.collections.tts.torch.g2p_classification_data import G2PClassificationDataset
from nemo.core.classes import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types import ChannelType, LogitsType, MaskType, NeuralType
from nemo.utils import logging

__all__ = ['G2PClassificationModel']


class G2PClassificationModel(ModelPT):
    """
    Transformer-based (duplex) tagger model for TN/ITN.
    """

    # @proper  lf

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer, add_prefix_space=True)
        self.max_sequence_len = cfg.get('max_sequence_len', self.tokenizer.model_max_length)
        self.wordids = None
        super().__init__(cfg=cfg, trainer=trainer)

        self.encoder = AutoModel.from_pretrained(cfg.encoder.transformer).encoder
        self.hidden_size = self.encoder.config.d_model
        self.classifier = SequenceClassifier(
            hidden_size=self.hidden_size,
            num_classes=2,
            num_layers=cfg.classifier_head.num_output_layers,
            activation='relu',
            log_softmax=False,
            dropout=cfg.classifier_head.fc_dropout,
            use_transformer_init=True,
            idx_conditioned_on=0,
        )

        # Loss Functions
        self.loss = CrossEntropyLoss()

        # setup to track metrics
        self.classification_report = ClassificationReport(num_classes=2, mode='macro', dist_sync_on_step=True)

        # Language
        self.lang = cfg.get('lang', None)

    # @typecheck()
    def forward(self, input_ids, attention_mask):
        hidden_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]

        # [B, 1, hid-dim]
        sentence_embedding = torch.mean(hidden_states, dim=1).unsqueeze(1)
        logits = self.classifier(hidden_states=sentence_embedding)
        return logits

    # Training
    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        input_ids, attention_mask, targets = batch
        logits = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.loss(logits=logits, labels=targets)
        self.log('train_loss', loss)
        return loss

    def training_epoch_end(self, outputs):
        return super().training_epoch_end(outputs)

    # Validation and Testing
    def validation_step(self, batch, batch_idx, split="val"):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids, attention_mask, targets = batch
        logits = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        val_loss = self.loss(logits=logits, labels=targets)
        self.log(f"{split}_loss", val_loss)
        preds = torch.argmax(logits, axis=-1)
        tp, fn, fp, _ = self.classification_report(preds, targets)
        return {f'{split}_loss': val_loss, 'tp': tp, 'fn': fn, 'fp': fp}

    def validation_epoch_end(self, outputs, split="val"):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x[f'{split}_loss'] for x in outputs]).mean()

        # calculate metrics and classification report
        precision, recall, f1, report = self.classification_report.compute()

        logging.info(f'{split}_report: {report}')

        self.log(f'{split}_loss', avg_loss, prog_bar=True)
        self.log(f'{split}_precision', precision)
        self.log(f'{split}_f1', f1)
        self.log(f'{split}_recall', recall)

        self.classification_report.reset()

    def test_step(self, batch, batch_idx):
        """
        Lightning calls this inside the test loop with the data from the test dataloader
        passed in as `batch`.
        """
        return self.validation_step(batch, batch_idx, "test")

    def test_epoch_end(self, outputs):
        """
        Called at the end of test to aggregate outputs.
        :param outputs: list of individual outputs of each test step.
        """
        return self.validation_epoch_end(outputs, "test")

    # Functions for inference
    @torch.no_grad()
    def _infer(self, sents: List[List[str]], inst_directions: List[str]):
        pass

    # Functions for processing data
    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        if not train_data_config or train_data_config.dataset.data_dir is None:
            logging.info(
                f"Dataloader config or file_path for the train is missing, so no data loader for train is created!"
            )
            self._train_dl = None
            return
        self._train_dl = self._setup_dataloader_from_config(cfg=train_data_config, data_split="train")

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        if not val_data_config or val_data_config.dataset.data_dir is None:
            logging.info(
                f"Dataloader config or file_path for the validation is missing, so no data loader for validation is created!"
            )
            self._validation_dl = None
            return
        self._validation_dl = self._setup_dataloader_from_config(cfg=val_data_config, data_split="val")

    def setup_test_data(self, test_data_config: Optional[DictConfig]):
        if not test_data_config or test_data_config.dataset.data_dir is None:
            logging.info(
                f"Dataloader config or file_path for the test is missing, so no data loader for test is created!"
            )
            self._test_dl = None
            return
        self._test_dl = self._setup_dataloader_from_config(cfg=test_data_config, data_split="test")

    def _setup_dataloader_from_config(self, cfg: DictConfig, data_split: str):
        if "dataset" not in cfg or not isinstance(cfg.dataset, DictConfig):
            raise ValueError(f"No dataset for {data_split}")
        if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
            raise ValueError(f"No dataloader_params for {data_split}")

        if data_split == "train":
            self.wordids = cfg.dataset.wordids

        if self.wordids is None or not os.path.exists(self.wordids):
            raise ValueError(f"Wordids failed to setup or doesn't exist - {data_split}")

        dataset = G2PClassificationDataset(
            dir_name=cfg.dataset.data_dir,
            tokenizer=self.tokenizer,
            wordid_map=self.wordids,
            context_len=cfg.dataset.context_len,
            max_seq_len=self.max_sequence_len,
            with_labels=True,
        )

        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)

    def input_example(self):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        sample = next(self.parameters())
        input_ids = torch.randint(low=0, high=2048, size=(2, 16), device=sample.device)
        attention_mask = torch.randint(low=0, high=1, size=(2, 16), device=sample.device)
        return tuple([input_ids, attention_mask])

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.
        Returns:
            List of available pre-trained models.
        """
        result = []
        result.append(
            PretrainedModelInfo(
                pretrained_model_name="neural_text_normalization_t5",
                location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/neural_text_normalization_t5/versions/1.5.0/files/neural_text_normalization_t5_tagger.nemo",
                description="Text Normalization model's tagger model.",
            )
        )
        return result
