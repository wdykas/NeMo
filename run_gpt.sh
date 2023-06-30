#!/bin/bash

# Parameters
#SBATCH --dependency=singleton
#SBATCH --exclusive
#SBATCH --gpus-per-node=4
#SBATCH --job-name=nemo-megatron-gpt3_5b
#SBATCH --mem=0
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --partition=asus_a100x4_40gb
#SBATCH --time=01:00:00

# setup
export TRANSFORMER_OFFLINE=1

# command 1
srun --container-image nvcr.io/ea-bignlp/nemofw-training:23.05-py3 --container-mounts /home/u00u4x8p3enW0rzLCW357/gpt_dataset:/home/u00u4x8p3enW0rzLCW357/gpt_dataset,/home/u00u4x8p3enW0rzLCW357/teconversion/NeMo:/opt/NeMo --no-container-mount-home bash -c "TRANSFORMERS_OFFLINE=1 HYDRA_FULL_ERROR=1 NCCL_DEBUG=INFO cd /opt/NeMo;
  git rev-parse HEAD;
  export PYTHONPATH=/opt/NeMo:\${PYTHONPATH};
  CUDA_DEVICE_MAX_CONNECTIONS=1 python3 -u /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
	  trainer.precision=bf16 \
	  trainer.num_nodes=1 \
	  trainer.devices=4 \
	  model.tensor_model_parallel_size=2 \
	  model.pipeline_model_parallel_size=2 \
	  model.transformer_engine=True \
	  model.global_batch_size=64 \
	  model.micro_batch_size=1 \
	  model.use_flash_attention=False \
	  model.tokenizer.merge_file=/home/u00u4x8p3enW0rzLCW357/gpt_dataset/bpe/merges.txt \
	  model.tokenizer.vocab_file=/home/u00u4x8p3enW0rzLCW357/gpt_dataset/bpe/vocab.json \
	  model.data.data_prefix=[1.0,/home/u00u4x8p3enW0rzLCW357/gpt_dataset/Wikipedia_en_ftfy_id_shuf_text_document]"

