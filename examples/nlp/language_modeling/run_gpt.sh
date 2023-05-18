TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 torchrun --nnode 1 --nproc_per_node 4 megatron_gpt_pretraining.py \
    trainer.devices=4 \
    model.data.data_prefix=[1,/workspace/gpt-wiki/Wikipedia_en_ftfy_id_shuf_text_document] \
    model.tensor_model_parallel_size=2 \
    model.pipeline_model_parallel_size=2 \
    model.micro_batch_size=4 \
    model.global_batch_size=16 \
    model.data.dataloader_type=single \
    model.seed=32345435 \
    model.tokenizer.vocab_file=/workspace/gpt-wiki/gpt2-vocab.json \
    model.tokenizer.merge_file=/workspace/gpt-wiki/gpt2-merges.txt
