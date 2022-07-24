tokens_in_batch="$1"
ds_name=commercial
python ~/NeMo/examples/nlp/token_classification/data/create_punctuation_capitalization_tarred_dataset.py \
  --text "/media/apeganov/DATA/punctuation_and_capitalization/simplest/3_128/${ds_name}/train/input.txt" \
  --labels "/media/apeganov/DATA/punctuation_and_capitalization/simplest/3_128/${ds_name}/train/bert_labels.txt" \
  --output_dir "/media/apeganov/DATA/punctuation_and_capitalization/simplest/3_128/${ds_name}/for_upload/train_bert_tarred_${tokens_in_batch}" \
  --tokens_in_batch="${tokens_in_batch}" \
  --num_batches_per_tarfile 100 \
  --lines_per_dataset_fragment 200000 \
  --tokenizer_name /home/apeganov/pretrained_tokenizers/bert_base_uncased \
  --use_fast_tokenizer
