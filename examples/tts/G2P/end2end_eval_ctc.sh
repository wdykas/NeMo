#!/bin/bash



JOB_ID=$1
BATCH_SIZE=32
PER_WORD=${2:-"None"}
VERSION=8

if [[ ${PER_WORD,,} == "true" ]]; then
  PER_WORD="clean_word_level=True"
  else PER_WORD=""
fi
echo "PER_WORD: ${PER_WORD}"

for file in /mnt/sdb_4/g2p/chpts/conformer/v${VERSION}/${JOB_ID}/g2p/G2PCTC/*/checkpoints/G2PCTC.nemo
do
  LOG_CLEAN=${JOB_ID}_log_clean.txt
  LOG_NO_CLEAN=${JOB_ID}_log_no_clean.txt

#  # eval with --CLEAN
#  output_heteronyms="CTC_${JOB_ID}.json"
#  python ctc_g2p_inference.py pretrained_model=$file manifest_filepath=/mnt/sdb_4/g2p/data_ipa/evaluation_sets_v${VERSION}/CMU_TEST_MULTI/cmu_test.json batch_size=$BATCH_SIZE > "/tmp/cmu_clean.txt"
#  python ctc_g2p_inference.py pretrained_model=$file manifest_filepath=/mnt/sdb_4/g2p/data_ipa/evaluation_sets_v${VERSION}/eval_wikihomograph.json output_file=$output_heteronyms batch_size=$BATCH_SIZE $PER_WORD > "/tmp/wiki_clean.txt"
#  python evaluate_heteronyms.py --manifest=$output_heteronyms > "/tmp/wiki_heteronyms_clean.txt"
#  tail -n 5 "/tmp/cmu_clean.txt" "/tmp/wiki_clean.txt" "/tmp/wiki_heteronyms_clean.txt" > ${LOG_CLEAN}

  output_heteronyms_wiki="ctc_generative_${JOB_ID}_wiki.json"
  output_heteronyms_lj_hifi_dev="ctc_generative_${JOB_ID}_lj_hifi_dev.json"
  python g2p_ctc_inference.py pretrained_model=$file manifest_filepath=/mnt/sdb_4/g2p/data_ipa/evaluation_sets_v${VERSION}/CMU_TEST_MULTI/cmu_test.json batch_size=$BATCH_SIZE > "/tmp/cmu${JOB_ID}.txt"
  python g2p_ctc_inference.py pretrained_model=$file manifest_filepath=/mnt/sdb_4/g2p/data_ipa/evaluation_sets_v${VERSION}/eval_wikihomograph.json output_file=$output_heteronyms_wiki batch_size=$BATCH_SIZE > "/tmp/wiki${JOB_ID}.txt"
  python evaluate_heteronyms.py --manifest=$output_heteronyms_wiki > "/tmp/wiki_heteronyms${JOB_ID}.txt"
  python g2p_ctc_inference.py pretrained_model=$file manifest_filepath=/mnt/sdb_4/g2p/data_ipa/disambiguated/ipa/100entries_92_train_lj_hifi_dev_jocelyn_short_ipa.json output_file=$output_heteronyms_lj_hifi_dev batch_size=$BATCH_SIZE > "/tmp/lj_hifi_dev${JOB_ID}.txt"
  python evaluate_heteronyms.py --manifest=$output_heteronyms_lj_hifi_dev > "/tmp/lj_hifi_dev_heteronyms${JOB_ID}.txt"
  tail -n 7 "/tmp/cmu${JOB_ID}.txt" "/tmp/wiki${JOB_ID}.txt" "/tmp/wiki_heteronyms${JOB_ID}.txt" "/tmp/lj_hifi_dev_heteronyms${JOB_ID}.txt" > ${LOG_NO_CLEAN}

#  echo "=======> Eval for ${JOB_ID} with CLEAN=True"
#  cat ${LOG_CLEAN}
  echo "=======> Eval for ${JOB_ID} with CLEAN=False"
  cat ${LOG_NO_CLEAN}
done