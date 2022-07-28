set -x
WANDB_API_KEY="$1"
exp_name="$2"
gpus="$3"

set -e -x
result_dir=/result/nemo_experiments
mkdir -p "${result_dir}"
cd /workspace/NeMo
git checkout feat/punc_tarred
git pull
bash reinstall.sh
cd examples/nlp/token_classification
wandb login ${WANDB_API_KEY}
python punctuation_capitalization_train_evaluate.py --config-path=conf \
    --config-name commercial_bs320k_steps400k \
    exp_manager.wandb_logger_kwargs.name=${exp_name} \
    +exp_manager.explicit_log_dir="${result_dir}" \
    trainer.gpus=${gpus} \
    model.language_model.pretrained_model_name=distilbert-base-uncased \
    model.train_ds.ds_item=/data/train_bert_tarred_10000 \
    model.validation_ds.ds_item=[/data/europarl_segments_dev,\
/data/europarl_sentences_dev,\
/data/google_segments_dev,\
/data/google_sentences_dev,\
/data/pg19_segments_dev,\
/data/pg19_sentences_dev,\
/data/pubmed_segments_dev,\
/data/pubmed_sentences_dev,\
/data/tatoeba_segments_dev,\
/data/tatoeba_sentences_dev,\
/data/un_segments_dev,\
/data/un_sentences_dev] \
    model.test_ds.ds_item=[/data/europarl_segments_test,\
/data/europarl_sentences_test,\
/data/google_segments_test,\
/data/google_sentences_test,\
/data/pg19_segments_test,\
/data/pg19_sentences_test,\
/data/pubmed_segments_test,\
/data/pubmed_sentences_test,\
/data/tatoeba_segments_test,\
/data/tatoeba_sentences_test,\
/data/un_segments_test,\
/data/un_sentences_test]
set +x
