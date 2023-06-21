JOBLIB_TEMP_FOLDER=/tmp TRANSFORMERS_OFFLINE=1 HYDRA_FULL_ERROR=1 torchrun --nnode 1 --nproc_per_node 8 megatron_gpt_pretraining.py \
    trainer.devices=8 \
    model.data.data_prefix=[1,/workspace/gpt-data/Wikipedia_en_ftfy_id_shuf_text_document] \
    model.tensor_model_parallel_size=2 \
    model.pipeline_model_parallel_size=2 \
    model.micro_batch_size=8 \
    model.global_batch_size=64 \
    model.data.dataloader_type=single \
    model.seed=32345435 \
    model.tokenizer.vocab_file=/workspace/gpt-data/gpt2-vocab.json \
    model.tokenizer.merge_file=/workspace/gpt-data/gpt2-merges.txt \
    model.resume_from_checkpoint="s3://wdykas-nemo-s3-gptcheckpoint/megatron_gpt/checkpoints/megatron_gpt--val_loss-5.26-step-800-consumed_samples-51136.0-last.ckpt" \
    exp_manager.explicit_log_dir="s3://wdykas-nemo-s3-gptcheckpoint/"
    #model.resume_from_checkpoint="s3://wdykas-gpt-checkpoints-extra-s3/checkpoints/megatron_gpt--val_loss\=8.31-step\=100-consumed_samples\=6336.0.ckpt"
    #model.resume_from_checkpoint="s3://wdykas-nemo-s3-gptcheckpoint/megatron_gpt/checkpoints/megatron_gpt--val_loss-5.26-step-800-consumed_samples-51136.0-last.ckpt" \
