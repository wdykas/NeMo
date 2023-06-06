NCCL_DEBUG=INFO torchrun --nnode 1 --nproc_per_node 1 megatron_gpt_pretraining.py \
    trainer.devices=1 \
    model.data.data_prefix=[1,/gpt-data/Wikipedia_en_ftfy_id_shuf_text_document] \
    model.tensor_model_parallel_size=1 \
    model.pipeline_model_parallel_size=1 \
    model.micro_batch_size=8 \
    model.global_batch_size=64 \
    model.data.dataloader_type=single \
    model.seed=32345435 \
    model.tokenizer.vocab_file=/gpt-data/gpt2-vocab.json \
    model.tokenizer.merge_file=/gpt-data/gpt2-merges.txt
~                                                
