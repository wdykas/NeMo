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
import logging
import copy
from operator import itemgetter
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from apex.transformer import tensor_parallel
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer

from nemo.collections.nlp.data.language_modeling.t0_task_manager import get_task_name
from nemo.collections.nlp.data.language_modeling.t0_dataset import (
    T0DatasetBuilder,
    T0PrimeDatasetBuilder,
    T0HFDatasetBuilder,
    T0PrimeHFDatasetBuilder,
    T0SSLPrimeDatasetBuilder,
    T0SSLPrimeHFDatasetBuilder
)
from nemo.collections.common.metrics.classification_accuracy import ExactStringPerCategoryMatchMetric
from nemo.collections.nlp.models.language_modeling.megatron_glue_model import MegatronT5FineTuneModel
from nemo.collections.nlp.modules.common.prompt_encoder import PromptEncoder
from nemo.collections.nlp.modules.common.t0sslprime_module import SeqProjectionNetwork, SSLPromptLoss

from nemo.collections.nlp.modules.common.megatron.utils import (
    average_losses_across_data_parallel_group,
    build_position_ids
)
from nemo.utils import logging


try:
    from apex.transformer import parallel_state
    from apex.transformer.pipeline_parallel.schedules.common import _get_params_for_weight_decay_optimization

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False


class MegatronT0Model(MegatronT5FineTuneModel):
    """
    Megatron t0 multitask fine tuning model
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg=cfg, trainer=trainer)
        self.cfg = cfg
        self.decoder_seq_length = cfg.get('decoder_seq_length', 10)
        self.acc_metric_dict = {}
        self._reduced_loss_buffer = []

    def get_loss(self, batch):
        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, task_ids, \
        prompt_ids, ctx_ex_prompt, ctx_ex_mask \
            = self.process_batch(batch)
        tokens_loss = itemgetter("tokens_loss")(self.model(
            encoder_input_ids=tokens_enc, decoder_input_ids=tokens_dec,
            encoder_attn_mask=enc_mask, decoder_attn_mask=dec_mask,
            tokentype_ids=None, lm_labels=labels
        ))
        loss = self.model.loss_func(loss_mask, tokens_loss)
        return loss

    def training_step(self, batch, batch_idx):

        loss = self.get_loss(batch)
        # Reduced loss for logging. Averages the loss across all workers unlike above which is specific to a DDP rank.
        reduced_loss = average_losses_across_data_parallel_group([loss])
        # cache reduced loss while accumulating gradients
        self._reduced_loss_buffer.append(reduced_loss[0])

        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            # Reduced loss for logging.
            average_reduced_loss = sum(self._reduced_loss_buffer) / len(self._reduced_loss_buffer)
            self.log('reduced_train_loss', average_reduced_loss, prog_bar=True)
            lr = self._optimizer.param_groups[0]['lr']
            self.log('lr', lr)
            self.log('global_step', self.trainer.global_step, prog_bar=True)
            self._reduced_loss_buffer = []
        return loss

    def train_dataloader(self):
        if self.cfg.data.num_data_shards > 1 and 0 < self.trainer.current_epoch < self.trainer.max_epochs:
            current_shard = self.trainer.current_epoch % self.cfg.data.num_data_shards
            self.setup_training_data(self.cfg.data, shard=current_shard)
        return self._train_dl

    def get_accuracy(self, predicted_token_ids, labels, task_ids, prompt_ids):
        preds = predicted_token_ids.cpu().numpy().tolist()
        labels = labels.cpu().numpy().tolist()
        for i, (pred, label, task_id, prompt_id) in enumerate(zip(preds, labels, task_ids, prompt_ids)):
            if self.model.tokenizer.eos_id in pred:
                idx = pred.index(self.model.tokenizer.eos_id)
                pred = pred[:idx]
            if hasattr(self.model.tokenizer, 'special_token_to_id'):
                pred = [id for id in pred if id not in self.model.tokenizer.additional_special_tokens_ids]
                label = [id for id in label if id not in self.model.tokenizer.additional_special_tokens_ids]
            pred = self.model.tokenizer.ids_to_text(pred)
            label = self.model.tokenizer.ids_to_text(label)

            task_id = task_id.item()
            prompt_id = prompt_id.item()
            _ = self.acc_metric_dict[task_id][prompt_id](pred, label)

    def inference_step(self, batch, batch_idx):
        loss = self.model.validation_step(batch, batch_idx)

        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, task_ids, \
        prompt_ids, ctx_ex_prompt, ctx_ex_mask \
            = self.process_batch(batch)

        predicted_token_ids, log_probs = self.model.decode(
            tokens_enc=tokens_enc, enc_mask=enc_mask,
            num_tokens_to_generate=self.decoder_seq_length
        )
        self.get_accuracy(predicted_token_ids, labels, task_ids, prompt_ids)

        return {'loss': loss}

    def inference_epoch_end(self, outputs_list, task_names):
        """Uses exact match"""
        accuracies_losses = {}
        for task_name, outputs in zip(task_names, outputs_list):
            loss = [x['loss'] for x in outputs]
            averaged_loss = average_losses_across_data_parallel_group(loss)
            accuracies_losses[task_name] = {'loss': averaged_loss[0]}
        for task_id in self.acc_metric_dict.keys():
            task_name = get_task_name(task_id)
            assert task_name in accuracies_losses
            accuracies_losses[task_name]['accuracies'] = {}
            for prompt_id in self.acc_metric_dict[task_id].keys():
                accuracy = self.acc_metric_dict[task_id][prompt_id].compute()['acc']
                if torch.any(torch.isnan(accuracy)):
                    continue
                accuracies_losses[task_name]['accuracies'][str(prompt_id)] = accuracy
                self.acc_metric_dict[task_id][prompt_id].reset()
        return accuracies_losses

    def validation_step(self, batch, batch_idx, dataloader_idx):
        return self.inference_step(batch, batch_idx)

    def validation_epoch_end(self, outputs_list):
        avg_loss = []
        avg_val_acc = []
        val_accuracies_losses = self.inference_epoch_end(outputs_list, self.validation_task_names)
        for task_name in val_accuracies_losses.keys():
            loss = val_accuracies_losses[task_name]['loss']
            avg_loss.append(loss)
            self.log(f'val_loss_{task_name}', loss, prog_bar=True)
            logging.info(f'Validation loss for {task_name}: {loss}')
            accuracies = val_accuracies_losses[task_name]['accuracies']
            for prompt_id in accuracies.keys():
                self.log(f'validation_acc_{task_name}_{prompt_id}', accuracies[prompt_id], prog_bar=False)
            avg_task_acc_list = list(accuracies.values())
            avg_task_acc = torch.mean(torch.stack(avg_task_acc_list))
            self.log(f'validation_acc_{task_name}', avg_task_acc, prog_bar=True)
            logging.info(f'Validation accuracy for {task_name}: {avg_task_acc}')
            avg_val_acc.extend(avg_task_acc_list)
        self.log('val_loss', torch.mean(torch.stack(avg_loss)), prog_bar=True)
        self.log('val_acc', torch.mean(torch.stack(avg_val_acc)), prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx):
        return self.inference_step(batch, batch_idx)

    def test_epoch_end(self, outputs_list):
        avg_test_acc = []
        test_accuracies_losses = self.inference_epoch_end(outputs_list, self.test_task_names)
        for task_name in test_accuracies_losses.keys():
            loss = test_accuracies_losses[task_name]['loss']
            self.log(f'test_loss_{task_name}', loss, prog_bar=True)
            logging.info(f'Test loss for {task_name}: {loss}')
            accuracies = test_accuracies_losses[task_name]['accuracies']
            for prompt_id in accuracies.keys():
                self.log(f'test_acc_{task_name}_{prompt_id}', accuracies[prompt_id], prog_bar=False)
            avg_task_acc_list = list(accuracies.values())
            avg_task_acc = torch.mean(torch.stack(avg_task_acc_list))
            self.log(f'test_acc_{task_name}', avg_task_acc, prog_bar=True)
            logging.info(f'Test accuracy for {task_name}: {avg_task_acc}')
            avg_test_acc.extend(avg_task_acc_list)
        self.log('test_acc', torch.mean(torch.stack(avg_test_acc)), prog_bar=True)

    def get_datasetbuilder(self, split, seq_length):
        if not getattr(self.cfg.data, 'use_hf_data', False):
            # Use NeMo dataset
            datasetbuilder = T0DatasetBuilder(
                t0_type=self.cfg.data.t0_type,
                dir_path=self.cfg.data.dir_path,
                max_sampling_size=self.cfg.data.max_sampling_size,
                split=split,
                tokenizer=self.model.tokenizer,
                max_seq_length=seq_length,
                seed=self.cfg.seed,
                use_cache=self.cfg.data.use_cache,
                max_samples=getattr(self.cfg.data, "max_samples", None),
                num_proc=self.cfg.data.num_workers,
                num_nodes=self.trainer.num_nodes,
                num_gpus=self.trainer.gpus,
                num_data_shards=self.cfg.data.num_data_shards,
                num_in_context_ex=getattr(self.cfg.data, "num_in_context_ex", None)
            )
        else:
            # Use Huggingface dataset
            datasetbuilder = T0HFDatasetBuilder(
                t0_type=self.cfg.data.t0_type,
                dir_path=self.cfg.data.dir_path,
                max_sampling_size=self.cfg.data.max_sampling_size,
                split=split,
                tokenizer=self.model.tokenizer,
                max_seq_length=seq_length,
                seed=self.cfg.seed,
                use_cache=self.cfg.data.use_cache,
                max_samples=getattr(self.cfg.data, "max_samples", None),
                num_proc=self.cfg.data.num_workers,
                num_nodes=self.trainer.num_nodes,
                num_gpus=self.trainer.gpus,
                distribute_datasets = getattr(self.cfg.data, "distribute_datasets", False),
                num_in_context_ex= getattr(self.cfg.data, "num_in_context_ex", None)
            )
        return datasetbuilder

    def build_data_loader(self, dataset, collate_fn, batch_size, shuffle, pin_memory):
        """Buld dataloader given an input dataset."""
        if dataset is None:
            return None

        #generator = torch.Generator()
        #generator.manual_seed(self.cfg.seed + (10 * self.trainer.local_rank))
        #sampler = torch.utils.data.sampler.RandomSampler(
        #    data_source=dataset, generator=generator
        #)

        rank = parallel_state.get_data_parallel_rank()
        world_size = parallel_state.get_data_parallel_world_size()
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
        )
        sampler.set_epoch(self.trainer.current_epoch)
        return DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.cfg.data.num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )

    def create_acc_metric_dict(self, dataset):
        task_id = dataset.task.task_id
        if not getattr(self.cfg.data, 'use_hf_data', False):
            prompt_ids = [int(n.split('_')[-1].strip()) for n in dataset.features[0].keys() if 'prompt_id' in n]
        else:
            prompt_ids = [int(n.split('_')[-1].strip()) for n in dataset.data.column_names if 'prompt_id' in n]
        if not self.acc_metric_dict.__contains__(task_id):
            self.acc_metric_dict[task_id] = {}
        self.acc_metric_dict[task_id].update({
            prompt_id: ExactStringPerCategoryMatchMetric().to(self.device)
            for prompt_id in prompt_ids
        })

    def on_sanity_check_start(self):
        for vdl in self._validation_dl:
            self.create_acc_metric_dict(vdl.dataset)

    def on_test_epoch_start(self):
        for tdl in self._test_dl:
            self.create_acc_metric_dict(tdl.dataset)

    def setup(self, stage: Optional[str] = None):
        """A PTL method to setup the training, validation and test datasets."""
        if stage == 'predict':
            return
        if self._train_dl is not None and self._validation_dl is not None:
            return
        if stage in (None, 'fit'):
            logging.info('Building train %s datasets.' % self.cfg.data.t0_type)
            self.setup_training_data(self.cfg.data)
            logging.info('Building validation datasets.')
            self.setup_validation_data(self.cfg.data)
            logging.info(f'Finished building T0 train and evaluation datasets.')
        if stage in (None, 'test'):
            logging.info('Building test datasets.')
            self.setup_test_data(self.cfg.data)

    def setup_training_data(self, cfg, shard=None):
        if shard is None:
            self._train_ds = self.get_datasetbuilder('train', self.cfg.data.train_ds.max_seq_length)
        else:
            logging.info('Resetting train dataset for shard = %s' % shard)
            self._train_ds.set_data_dict(shard)

        logging.info(f'Length of train dataset: {len(self._train_ds)}')
        if hasattr(self, '_train_ds'):
            self._train_dl = self.build_data_loader(
                self._train_ds.assemble_datasets(),
                collate_fn=self._train_ds.collate_fn,
                batch_size=self.cfg.data.train_ds.batch_size,
                shuffle=True,
                pin_memory=True
            )

    def setup_validation_data(self, cfg):
        self._validation_ds = self.get_datasetbuilder('validation', self.cfg.data.validation_ds.max_seq_length)
        logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        if hasattr(self, '_validation_ds'):
            dataset_dict = self._validation_ds.assemble_datasets()
            self.validation_task_names = list(dataset_dict.keys())
            self._validation_dl = [self.build_data_loader(
                dataset,
                collate_fn=self._validation_ds.collate_fn,
                batch_size=self.cfg.data.validation_ds.batch_size,
                shuffle=False,
                pin_memory=True
            ) for dataset in dataset_dict.values()]

    def setup_test_data(self, cfg):
        self._test_ds = self.get_datasetbuilder('test', self.cfg.data.test_ds.max_seq_length)
        logging.info(f'Length of test dataset: {len(self._test_ds)}')
        if hasattr(self, '_test_ds'):
            dataset_dict = self._test_ds.assemble_datasets()
            self.test_task_names = list(dataset_dict.keys())
            self._test_dl = [self.build_data_loader(
                dataset,
                collate_fn=self._test_ds.collate_fn,
                batch_size=self.cfg.data.test_ds.batch_size,
                shuffle=False,
                pin_memory=True
            ) for dataset in dataset_dict.values()]

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass

    def process_batch(self, batch):
        """Build the batch."""

        keys = [
            'text_enc', 'text_dec', 'labels',
            'loss_mask', 'enc_mask', 'dec_mask',
            'task_ids', 'prompt_ids'
        ]
        if getattr(self.cfg.data, "num_in_context_ex", None) is not None:
            keys.append('ctx_ex_prompt')
            keys.append('ctx_ex_mask')
        datatype = torch.int64

        data = batch
        data_b = tensor_parallel.broadcast_data(keys, data, datatype)

        # Unpack.
        tokens_enc = data_b['text_enc'].long()
        tokens_dec = data_b['text_dec'].long()
        labels = data_b['labels'].long()
        loss_mask = data_b['loss_mask'].float()

        enc_mask = data_b['enc_mask']
        dec_mask = data_b['dec_mask']

        # T0 specific
        task_ids = data_b['task_ids'].long()
        prompt_ids = data_b['prompt_ids'].long()

        if getattr(self.cfg.data, "num_in_context_ex", None) is not None:
            ctx_ex_prompt = data_b['ctx_ex_prompt'].long()
            ctx_ex_mask = data_b['ctx_ex_mask'].float()
        else:
            ctx_ex_prompt = ctx_ex_mask = None

        return tokens_enc, tokens_dec, loss_mask, labels, \
               enc_mask, dec_mask, task_ids, prompt_ids, ctx_ex_prompt, ctx_ex_mask


class MegatronT0PrimeModel(MegatronT0Model):
    """
        Megatron t0 prime multitask fine tuning model using differentiable promtps
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg=cfg, trainer=trainer)

        self.hidden_size = self.model.cfg.hidden_size
        self.prompt_seq_len = cfg.prompt_encoder.seq_len
        self.float_type = self.model.enc_dec_model.enc_dec_model.encoder.model.layers[0].dtype

        if cfg.freeze:
            self.model.freeze()

        self.prompt_encoder = PromptEncoder(
            prompt_seq_len=self.prompt_seq_len,
            hidden_size=self.hidden_size,
            prompt_dropout=cfg.prompt_encoder.dropout,
            num_layers=cfg.prompt_encoder.num_layers,
            reparametrize=cfg.prompt_encoder.reparametrize,
            prompt_gen_type=cfg.prompt_encoder.prompt_gen_type,
            precision=cfg.precision,
            trainer=trainer
        )
        self.word_embeddings = self.model.enc_dec_model.encoder_embedding.word_embeddings
        self.position_embeddings = self.model.enc_dec_model.encoder_embedding.position_embeddings
        self.model.tokenizer.add_special_tokens({'additional_special_tokens': [cfg.pseudo_token]})
        self.pseudo_token_id = self.model.tokenizer.token_to_id(cfg.pseudo_token)
        self.pad_token_id = self.model.tokenizer.pad_id if self.model.tokenizer.pad_id is not None \
            else self.model.tokenizer.unk_id

    def get_prompt_embeddings(self, enc_input_id, ctx_ex_prompt, enc_mask, ctx_ex_mask, prompt_encoder=None):
        queries_for_embedding = enc_input_id.clone()
        queries_mask = enc_mask
        if ctx_ex_prompt is not None:
            queries_for_embedding = torch.cat((queries_for_embedding, ctx_ex_prompt.clone()), dim=1)
            queries_mask = torch.cat((queries_mask, ctx_ex_mask), dim=1)
        queries_for_embedding[(queries_for_embedding == self.pseudo_token_id)] = self.pad_token_id
        query_embeddings = self.word_embeddings(queries_for_embedding)

        if self.cfg.prompt_encoder.task_dependent:
            prompt_condition = query_embeddings
            prompt_condition_mask = queries_mask
        else:
            prompt_condition = None
            prompt_condition_mask= None

        if self.float_type == torch.float32:
            if prompt_encoder is None:
                prompt_embeds = self.prompt_encoder(
                    prompt_condition=prompt_condition,
                    prompt_condition_mask=prompt_condition_mask
                )
            else:
                prompt_embeds = prompt_encoder(
                    prompt_condition=prompt_condition,
                    prompt_condition_mask=prompt_condition_mask
                )
        else:
            with torch.autocast(device_type="cuda", dtype=self.float_type):
                if prompt_encoder is None:
                    prompt_embeds = self.prompt_encoder(
                        prompt_condition=prompt_condition,
                        prompt_condition_mask=prompt_condition_mask
                    )
                else:
                    prompt_embeds = prompt_encoder(
                        prompt_condition=prompt_condition,
                        prompt_condition_mask=prompt_condition_mask
                    )
        return prompt_embeds, query_embeddings

    def embed_input(self, enc_input_id, ctx_ex_prompt, prompt_input_ids, enc_mask, ctx_ex_mask):
        """
        This method will replace the virtual tokens in the enc_input_id with
        embeddings calculated from `prompt_encoder`. If the `prompt_input_ids` is
        not None, the computed virtual token embeddings are depenedent on it.
        The virtual token placeholders has the token_id `self.pseudo_token_id`.
        params:
            enc_input_id: the input token ids
            ctx_ex_prompt: token ids of in-context examples
            prompt_input_ids: the task specific template token ids
        returns:
            the token embedding for the LM model.
        """
        bz, seq_len = enc_input_id.shape

        prompt_embeds, query_embeddings = self.get_prompt_embeddings(
            enc_input_id, ctx_ex_prompt, enc_mask, ctx_ex_mask
        )

        query_embeddings = query_embeddings[:, :seq_len, :].clone().type(self.float_type)
        encoder_position_ids = build_position_ids(enc_input_id)
        position_embeddings = self.position_embeddings(encoder_position_ids)

        encoder_input = torch.cat((prompt_embeds, query_embeddings + position_embeddings.type(self.float_type)), dim=1)
        enc_mask = torch.cat((torch.ones_like(prompt_embeds[:, :, 0]), enc_mask), dim=1)

        return prompt_embeds, encoder_input, enc_mask

    def get_loss(self, batch):
        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, task_ids, \
        prompt_ids, tokens_prompt, ctx_ex_prompt, ctx_ex_mask \
            = self.process_batch(batch)

        prompt_emb, encoder_input, enc_mask = self.embed_input(
            tokens_enc, ctx_ex_prompt, tokens_prompt, enc_mask, ctx_ex_mask
        )
        if self.float_type == torch.float32:
            tokens_loss = itemgetter("tokens_loss")(self.model.enc_dec_model(
                enc_input_ids=tokens_enc, dec_input_ids=tokens_dec,
                enc_attn_mask=enc_mask, dec_attn_mask=dec_mask,
                tokentype_ids=None, labels=labels,
                enc_input=encoder_input
            ))
        else:
            with torch.autocast(device_type="cuda", dtype=self.float_type):
                tokens_loss = itemgetter("tokens_loss")(self.model.enc_dec_model(
                    enc_input_ids=None, dec_input_ids=tokens_dec,
                    enc_attn_mask=enc_mask, dec_attn_mask=dec_mask,
                    tokentype_ids=None, labels=labels,
                    enc_input=encoder_input
                ))
        loss = self.model.loss_func(loss_mask, tokens_loss)
        return loss

    def inference_step(self, batch, batch_idx):
        tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, task_ids, \
        prompt_ids, tokens_prompt, ctx_ex_prompt, ctx_ex_mask \
            = self.process_batch(batch)
        prompt_emb, encoder_input, enc_mask = self.embed_input(
            tokens_enc, ctx_ex_prompt, tokens_prompt, enc_mask, ctx_ex_mask
        )
        loss = self.get_loss(batch)
        reduced_loss = average_losses_across_data_parallel_group([loss])
        predicted_token_ids, log_probs = self.model.decode(
            tokens_enc=tokens_enc, enc_mask=enc_mask, enc_input=encoder_input,
            num_tokens_to_generate=self.decoder_seq_length
        )
        self.get_accuracy(predicted_token_ids, labels, task_ids, prompt_ids)

        return {'loss': reduced_loss}

    def get_datasetbuilder(self, split, seq_length):
        if not getattr(self.cfg.data, 'use_hf_data', False):
            # Use NeMo dataset
            datasetbuilder = T0PrimeDatasetBuilder(
                t0_type=self.cfg.data.t0_type,
                dir_path=self.cfg.data.dir_path,
                max_sampling_size=self.cfg.data.max_sampling_size,
                split=split,
                tokenizer=self.model.tokenizer,
                max_seq_length=seq_length,
                seed=self.cfg.seed,
                use_cache=self.cfg.data.use_cache,
                max_samples=getattr(self.cfg.data, "max_samples", None),
                num_proc=self.cfg.data.num_workers,
                num_nodes=self.trainer.num_nodes,
                num_gpus=self.trainer.gpus,
                num_data_shards=self.cfg.data.num_data_shards,
                prompt_token_id=self.pseudo_token_id,
                prompt_seq_len=self.prompt_seq_len,
                split_template=self.cfg.data.split_template,
                num_in_context_ex=getattr(self.cfg.data, "num_in_context_ex", None)
            )
        else:
            # Use Huggingface dataset
            datasetbuilder = T0PrimeHFDatasetBuilder(
                t0_type=self.cfg.data.t0_type,
                dir_path=self.cfg.data.dir_path,
                max_sampling_size=self.cfg.data.max_sampling_size,
                split=split,
                tokenizer=self.model.tokenizer,
                max_seq_length=seq_length,
                seed=self.cfg.seed,
                use_cache=self.cfg.data.use_cache,
                max_samples=getattr(self.cfg.data, "max_samples", None),
                num_proc=self.cfg.data.num_workers,
                num_nodes=self.trainer.num_nodes,
                num_gpus=self.trainer.gpus,
                distribute_datasets=getattr(self.cfg.data, "distribute_datasets", False),
                prompt_token_id=self.pseudo_token_id,
                prompt_seq_len=self.prompt_seq_len,
                split_template=self.cfg.data.split_template,
                num_in_context_ex=getattr(self.cfg.data, "num_in_context_ex", None)
            )
        return datasetbuilder

    def process_batch(self, batch):
        """Build the batch."""

        keys = [
            'text_enc', 'text_dec', 'template', 'labels',
            'loss_mask', 'enc_mask', 'dec_mask',
            'task_ids', 'prompt_ids'
        ]
        if getattr(self.cfg.data, "num_in_context_ex", None) is not None:
            keys.append('ctx_ex_prompt')
            keys.append('ctx_ex_mask')
        datatype = torch.int64
        data = batch
        data_b = tensor_parallel.broadcast_data(keys, data, datatype)

        # Unpack.
        tokens_enc = data_b['text_enc'].long()
        tokens_dec = data_b['text_dec'].long()
        labels = data_b['labels'].long()
        loss_mask = data_b['loss_mask'].float()

        enc_mask = data_b['enc_mask']
        dec_mask = data_b['dec_mask']

        # T0 specific
        task_ids = data_b['task_ids'].long()
        prompt_ids = data_b['prompt_ids'].long()

        # T0' specific
        tokens_prompt = data_b['template'].long()

        if getattr(self.cfg.data, "num_in_context_ex", None) is not None:
            ctx_ex_prompt = data_b['ctx_ex_prompt'].long()
            ctx_ex_mask = data_b['ctx_ex_mask'].float()
        else:
            ctx_ex_prompt = ctx_ex_mask = None

        return tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, \
               task_ids, prompt_ids, tokens_prompt, ctx_ex_prompt, ctx_ex_mask


class MegatronT0SSLPrimeModel(MegatronT0PrimeModel):
    """
        Megatron t0 prime multitask fine tuning model using differentiable promtps and contrastive learning
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg=cfg, trainer=trainer)

        self.ssl_proj_student = SeqProjectionNetwork(
            hidden_size=self.hidden_size,
            num_virtual_classes=cfg.ssl.num_virtual_classes,  # perhaps = num_task * num_prompts?
            nlayers=cfg.ssl.nlayers,
            use_ln=cfg.ssl.use_ln
        )
        self.ssl_loss = SSLPromptLoss(
            num_virtual_classes=cfg.ssl.num_virtual_classes,
            teacher_temp_init=cfg.ssl.teacher_temp_init,
            teacher_temp_final=cfg.ssl.teacher_temp_final,
            epochs=trainer.max_epochs,
            student_temp=cfg.ssl.student_temp,
            center_momentum=cfg.ssl.center_momentum
        )
        self.ssl_proj_teacher = SeqProjectionNetwork(
            hidden_size=self.hidden_size,
            num_virtual_classes=cfg.ssl.num_virtual_classes,  # perhaps = num_task * num_prompts?
            nlayers=cfg.ssl.nlayers,
            use_ln=cfg.ssl.use_ln
        )
        self.ssl_proj_teacher.freeze()
        #self.prompt_encoder.attribute = list(self.prompt_encoder.attribute)?
        self.teacher_net = PromptEncoder(
            prompt_seq_len=self.prompt_seq_len,
            hidden_size=self.hidden_size,
            prompt_dropout=cfg.prompt_encoder.dropout,
            num_layers=cfg.prompt_encoder.num_layers,
            reparametrize=cfg.prompt_encoder.reparametrize,
            prompt_gen_type=cfg.prompt_encoder.prompt_gen_type,
            precision=cfg.precision,
            trainer=trainer
        )
        self.teacher_net.freeze()

    def on_before_zero_grad(self, optimizer):
        # Exponential Moving Average update of teacher network
        student_modules = [self.ssl_proj_student, self.prompt_encoder]
        teacher_modules = [self.ssl_proj_teacher, self.teacher_net]
        alpha = self.cfg.ssl.teacher_grad_momentum
        for student, teacher in zip(student_modules, teacher_modules):
            student_params = student.parameters()
            teacher_params = teacher.parameters()
            for sparam, tparam in zip(student_params, teacher_params):
                with torch.no_grad():
                    tparam *= alpha
                    tparam += (1 - alpha) * sparam

    def get_loss(self, batch):
        if self.trainer.training:
            tokens_enc_1, tokens_dec, loss_mask, labels, enc_mask_1, dec_mask, task_ids, \
            prompt_ids, tokens_prompt, ctx_ex_prompt_1, ctx_ex_mask_1, \
            tokens_enc_2, enc_mask_2, tokens_prompt_2, ctx_ex_prompt_2, ctx_ex_mask_2 \
                = self.process_batch(batch)
        else:
            tokens_enc_1, tokens_dec, loss_mask, labels, enc_mask_1, dec_mask, task_ids, \
            prompt_ids, tokens_prompt, ctx_ex_prompt_1, ctx_ex_mask_1 \
                = self.process_batch(batch)
        prompt_embeds, encoder_input, enc_mask = self.embed_input(
            tokens_enc_1, ctx_ex_prompt_1, tokens_prompt, enc_mask_1, ctx_ex_mask_1
        )

        if self.trainer.training:
            ssl_loss = self.compute_ssl_loss(
                prompt_embeds, tokens_enc_1, ctx_ex_prompt_1, tokens_enc_2, ctx_ex_prompt_2,
                enc_mask_1, enc_mask_2, ctx_ex_mask_1, ctx_ex_mask_2
            )
            self.log('ssl_loss', ssl_loss)

        if self.float_type == torch.float32:
            tokens_loss = itemgetter("tokens_loss")(self.model.enc_dec_model(
                enc_input_ids=None, dec_input_ids=tokens_dec,
                enc_attn_mask=enc_mask, dec_attn_mask=dec_mask,
                tokentype_ids=None, labels=labels,
                enc_input=encoder_input
            ))
        else:
            with torch.autocast(device_type="cuda", dtype=self.float_type):
                tokens_loss = itemgetter("tokens_loss")(self.model.enc_dec_model(
                    enc_input_ids=None, dec_input_ids=tokens_dec,
                    enc_attn_mask=enc_mask, dec_attn_mask=dec_mask,
                    tokentype_ids=None, labels=labels,
                    enc_input=encoder_input
                ))
        loss = self.model.loss_func(loss_mask, tokens_loss)
        if self.trainer.training:
            return loss + ssl_loss
        else:
            return loss

    def compute_ssl_loss(
            self,
            prompt_embeds,
            tokens_enc_1,
            ctx_ex_prompt_1,
            tokens_enc_2,
            ctx_ex_prompt_2,
            enc_mask,
            enc_mask_2,
            ctx_ex_mask,
            ctx_ex_mask_2
    ):

        prompt_embeds_s1 = prompt_embeds
        prompt_embeds_s2, _ = self.get_prompt_embeddings(
            tokens_enc_2, ctx_ex_prompt_2, enc_mask_2, ctx_ex_mask_2
        )
        student_out1 = self.ssl_proj_student(prompt_embeds_s1)
        student_out2 = self.ssl_proj_student(prompt_embeds_s2)

        with torch.no_grad():
            prompt_embeds_t1, _ = self.get_prompt_embeddings(
                tokens_enc_1, ctx_ex_prompt_1, enc_mask, ctx_ex_mask, self.teacher_net
            )
            prompt_embeds_t2, _ = self.get_prompt_embeddings(
                tokens_enc_2, ctx_ex_prompt_2, enc_mask_2, ctx_ex_mask_2, self.teacher_net
            )
            teacher_out1 = self.ssl_proj_teacher(prompt_embeds_t1)
            teacher_out2 = self.ssl_proj_teacher(prompt_embeds_t2)

        ssl_loss1 = self.ssl_loss(student_out1, teacher_out2, self.trainer.current_epoch)
        ssl_loss2 = self.ssl_loss(student_out2, teacher_out1, self.trainer.current_epoch)
        ssl_loss = 0.5 * (ssl_loss1 + ssl_loss2)
        self.ssl_loss.update_center(teacher_out1, teacher_out2)
        return ssl_loss

    def get_datasetbuilder(self, split, seq_length):
        if not getattr(self.cfg.data, 'use_hf_data', False):
            # Use NeMo dataset
            datasetbuilder = T0SSLPrimeDatasetBuilder(
                t0_type=self.cfg.data.t0_type,
                dir_path=self.cfg.data.dir_path,
                max_sampling_size=self.cfg.data.max_sampling_size,
                split=split,
                tokenizer=self.model.tokenizer,
                max_seq_length=seq_length,
                seed=self.cfg.seed,
                use_cache=self.cfg.data.use_cache,
                max_samples=getattr(self.cfg.data, "max_samples", None),
                num_proc=self.cfg.data.num_workers,
                num_nodes=self.trainer.num_nodes,
                num_gpus=self.trainer.gpus,
                num_data_shards=self.cfg.data.num_data_shards,
                prompt_token_id=self.pseudo_token_id,
                prompt_seq_len=self.prompt_seq_len,
                split_template=self.cfg.data.split_template,
                num_in_context_ex=getattr(self.cfg.data, "num_in_context_ex", None)
            )
        else:
            # Use Huggingface dataset
            datasetbuilder = T0SSLPrimeHFDatasetBuilder(
                t0_type=self.cfg.data.t0_type,
                dir_path=self.cfg.data.dir_path,
                max_sampling_size=self.cfg.data.max_sampling_size,
                split=split,
                tokenizer=self.model.tokenizer,
                max_seq_length=seq_length,
                seed=self.cfg.seed,
                use_cache=self.cfg.data.use_cache,
                max_samples=getattr(self.cfg.data, "max_samples", None),
                num_proc=self.cfg.data.num_workers,
                num_nodes=self.trainer.num_nodes,
                num_gpus=self.trainer.gpus,
                distribute_datasets=getattr(self.cfg.data, "distribute_datasets", False),
                prompt_token_id=self.pseudo_token_id,
                prompt_seq_len=self.prompt_seq_len,
                split_template=self.cfg.data.split_template,
                num_in_context_ex=getattr(self.cfg.data, "num_in_context_ex", None)
            )
        return datasetbuilder

    def process_batch(self, batch):
        """Build the batch."""
        if not self.trainer.training:
            return super().process_batch(batch)
        keys = [
            'text_enc', 'text_dec', 'template', 'labels',
            'loss_mask', 'enc_mask', 'dec_mask',
            'task_ids', 'prompt_ids',
            'ssl_text_enc', 'ssl_template', 'ssl_enc_mask', 'ssl_prompt_ids'
        ]
        if getattr(self.cfg.data, "num_in_context_ex", None) is not None:
            keys.append('ctx_ex_prompt')
            keys.append('ctx_ex_mask')
            keys.append('ssl_ctx_ex_prompt')
            keys.append('ssl_ctx_ex_mask')
        datatype = torch.int64
        data = batch
        data_b = tensor_parallel.broadcast_data(keys, data, datatype)

        # Unpack.
        tokens_enc = data_b['text_enc'].long()
        tokens_dec = data_b['text_dec'].long()
        labels = data_b['labels'].long()
        loss_mask = data_b['loss_mask'].float()
        tokens_enc_2 = data_b['ssl_text_enc'].long()

        enc_mask = data_b['enc_mask']
        dec_mask = data_b['dec_mask']
        enc_mask_2 = data_b['ssl_enc_mask']

        # T0 specific
        task_ids = data_b['task_ids'].long()
        prompt_ids = data_b['prompt_ids'].long()

        # T0' specific
        tokens_prompt = data_b['template'].long()
        tokens_prompt_2 = data_b['ssl_template'].long()

        if getattr(self.cfg.data, "num_in_context_ex", None) is not None:
            ctx_ex_prompt = data_b['ctx_ex_prompt'].long()
            ctx_ex_mask = data_b['ctx_ex_mask']
            ctx_ex_prompt_2 = data_b['ssl_ctx_ex_prompt'].long()
            ctx_ex_mask_2 = data_b['ssl_ctx_ex_mask']
        else:
            ctx_ex_prompt = ctx_ex_mask = ctx_ex_prompt_2 = ctx_ex_mask_2 = None
        return tokens_enc, tokens_dec, loss_mask, labels, enc_mask, dec_mask, \
               task_ids, prompt_ids, tokens_prompt, ctx_ex_prompt, ctx_ex_mask, \
               tokens_enc_2, enc_mask_2, tokens_prompt_2, ctx_ex_prompt_2, ctx_ex_mask_2