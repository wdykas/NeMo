TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=INFO HYDRA_FULL_ERROR=1 torchrun --nnode 1 --nproc_per_node 8 megatron_gpt_pretraining.py \
    trainer.devices=8 \
    model.data.data_prefix=[1,/workspace/gpt-data/Wikipedia_en_ftfy_id_shuf_text_document] \
    model.tensor_model_parallel_size=1 \
    model.micro_batch_size=4 \
    model.global_batch_size=64 \
    model.data.dataloader_type=single \
    model.seed=32345435 \
    model.tokenizer.vocab_file=/workspace/gpt-data/gpt2-vocab.json \
    model.tokenizer.merge_file=/workspace/gpt-data/gpt2-merges.txt \
    trainer.max_steps=1000 \
    exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True
