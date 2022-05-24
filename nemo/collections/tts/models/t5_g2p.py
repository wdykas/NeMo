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
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from transformers import T5ForConditionalGeneration, T5Tokenizer

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.tts.torch.data import T5G2PDataset
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.core.neural_types import LabelsType, LossType, MaskType, NeuralType, TokenIndex

__all__ = ['T5G2PModel']


@dataclass
class T5G2PConfig:
    train_ds: Optional[Dict[Any, Any]] = None
    validation_ds: Optional[Dict[Any, Any]] = None


class T5G2PModel(ModelPT):  # TODO: Check parent class
    """
    T5-based grapheme-to-phoneme model.
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "input_ids": NeuralType(('B', 'T'), TokenIndex()),
            "attention_mask": NeuralType(('B', 'T'), MaskType(), optional=True),
            "labels": NeuralType(('B', 'T'), LabelsType()),
        }

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {"loss": NeuralType((), LossType())}

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        self.world_size = 1
        if trainer is not None:
            self.world_size = trainer.num_nodes * trainer.num_gpus

        # Load appropriate tokenizer from HuggingFace
        self.model_name = cfg.get("model_name", "t5-small")  # One of: t5-small, t5-base, t5-large, t5-3b, t5-11b
        print(f"----------> Using model: {self.model_name}")
        self._tokenizer = T5Tokenizer.from_pretrained(self.model_name)

        self.max_source_len = cfg.get("max_source_len", self._tokenizer.model_max_length)
        self.max_target_len = cfg.get("max_target_len", self._tokenizer.model_max_length)

        # Ensure passed cfg is compliant with schema
        schema = OmegaConf.structured(T5G2PConfig)
        # ModelPT ensures that cfg is a DictConfig, but do this second check in case ModelPT changes
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        elif not isinstance(cfg, DictConfig):
            raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")
        OmegaConf.merge(cfg, schema)

        super().__init__(cfg, trainer)

        # Load pretrained T5 model from HuggingFace
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)

    @typecheck()
    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs.loss

    # ===== Training Functions ===== #
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        train_loss = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,)

        self.log('train_loss', train_loss)
        return train_loss

    def training_epoch_end(self, outputs):
        return super().training_epoch_end(outputs)

    # ===== Validation Functions ===== #
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch

        # Get loss from forward step
        val_loss = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels,)

        # Get preds from generate function and calculate PER
        labels_str = self._tokenizer.batch_decode(
            # Need to do the following to zero out the -100s (ignore_index).
            torch.ones_like(labels) * ((labels == -100) * 100) + labels,
            skip_special_tokens=True,
        )
        generated_str, _, _ = self._generate_predictions(input_ids=input_ids, model_max_target_len=self.max_target_len)
        per = word_error_rate(hypotheses=generated_str, references=labels_str)
        return {'val_loss': val_loss, 'per': per}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs (reduces across batches, not workers).
        """
        # TODO: Does this need a no_grad?
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss, sync_dist=True)

        # TODO: Add better PER calculation and logging.
        avg_per = sum([x['per'] for x in outputs]) / len(outputs)
        self.log('val_per', avg_per)
        print(f"---------------> PER: {avg_per}")

        return {'loss': avg_loss}

    @torch.no_grad()
    def _generate_predictions(self, input_ids: torch.Tensor, model_max_target_len: int = 512):
        """
        Generates predictions and converts IDs to text.
        """
        outputs = self.model.generate(
            input_ids, output_scores=True, return_dict_in_generate=True, max_length=model_max_target_len
        )

        generated_ids, sequence_toks_scores = outputs['sequences'], outputs['scores']
        generated_texts = self._tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_texts, generated_ids, sequence_toks_scores

    # ===== Dataset Setup Functions ===== #
    def _setup_dataloader_from_config(self, cfg, name):
        if "dataset" not in cfg or not isinstance(cfg.dataset, DictConfig):
            raise ValueError(f"No dataset for {name}")
        if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
            raise ValueError(f"No dataloader_params for {name}")

        dataset = T5G2PDataset(
            cfg.dataset.manifest_filepath, self._tokenizer, self.max_source_len, self.max_target_len
        )

        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)

    def setup_training_data(self, cfg):
        self._train_dl = self._setup_dataloader_from_config(cfg, name="train")

    def setup_validation_data(self, cfg):
        self._validation_dl = self._setup_dataloader_from_config(cfg, name="validation")

    def setup_test_data(self, cfg):
        self._test_dl = self._setup_dataloader_from_config(cfg, name="test")

    # ===== List Available Models - N/A =====$
    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        return []
