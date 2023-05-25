TORCH_DISTRIBUTED_DEBUG=DETAIL HYDRA_FULL_ERROR=1 torchrun --nnode 1 --nproc_per_node 2 megatron_t5_pretraining.py \
    trainer.devices=2 \
    model.data.data_prefix=[1,/workspace/t5-pile/my-t5_20_bert_tokenizer_text_document] \
    model.tensor_model_parallel_size=1 \
    model.micro_batch_size=4 \
    model.global_batch_size=16 \
    model.data.dataloader_type=single \
    model.seed=32345435 \
    model.tokenizer.vocab_file=/workspace/t5-pile/bert-large-cased-vocab.txt
