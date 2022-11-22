



CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 torchrun --nnode 1 --nproc_per_node 4 megatron_bert_pretraining.py \
    model.data.data_prefix=[1.0,/workspace/bert_pile5/my-bert-05-pile_text_sentence] \
    model.tensor_model_parallel_size=1 \
    model.pipeline_model_parallel_size=1 \
    model.micro_batch_size=4 \
    model.global_batch_size=512 \
    model.tokenizer.vocab_file=/workspace/bert_pile5/bert-large-cased-vocab.txt
