TRAIN="[/workspace/dolly-data/databricks-dolly-15k-output.jsonl]"

VALID="[/workspace/dolly-data/databricks-dolly-15k-output.jsonl]"

TEST="[/workspace/dolly-data/databricks-dolly-15k-output.jsonl]"
VALID_NAMES="[your-validation-dataset-name]"

CONCAT_SAMPLING_PROBS="[1.0]"

TP_SIZE=1

PP_SIZE=1

python /workspace/sft-nemo/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_sft.py \
  trainer.precision=16 \
  trainer.max_steps=1000 \
  trainer.devices=2 \
  trainer.val_check_interval=200 \
  model.megatron_amp_O2=False \
  model.tensor_model_parallel_size=${TP_SIZE} \
  model.pipeline_model_parallel_size=${PP_SIZE} \
  model.optim.lr=5e-6 \
  model.answer_only_loss=True \
  model.data.train_ds.micro_batch_size=1 \
  model.data.train_ds.global_batch_size=128 \
  model.data.train_ds.file_names=${TRAIN} \
  model.data.validation_ds.micro_batch_size=1 \
  model.data.validation_ds.global_batch_size=128 \
  model.data.validation_ds.file_names=${VALID} \
  model.data.test_ds.file_names=${TEST} \
  model.data.test_ds.micro_batch_size=1 \
  model.data.test_ds.global_batch_size=128 \
  model.data.train_ds.num_workers=0 \
  model.data.validation_ds.num_workers=0 \
  model.data.test_ds.num_workers=0 \
  model.data.validation_ds.metric.name=loss \
  model.data.test_ds.metric.name=loss \
  exp_manager.create_wandb_logger=True \
  exp_manager.explicit_log_dir=/results \
  exp_manager.resume_if_exists=True \
  exp_manager.resume_ignore_no_checkpoint=True \
  exp_manager.create_checkpoint_callback=True \
  exp_manager.checkpoint_callback_params.monitor=validation_loss \
  model.restore_from_path="/workspace/sft-nemo/NeMo/examples/nlp/language_modeling/nemo_experiments/megatron_gpt/checkpoints/megatron_gpt.nemo" \
  model.data.train_ds.concat_sampling_probabilities=${CONCAT_SAMPLING_PROBS}
  #model.data.validation_ds.names=${VALID_NAMES} \

