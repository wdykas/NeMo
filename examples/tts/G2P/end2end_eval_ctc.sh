#!/bin/bash



JOB_ID=$1
BATCH_SIZE=32
PER_WORD=${2:-"None"}
VERSION=7

if [[ ${PER_WORD,,} == "true" ]]; then
  PER_WORD="clean_word_level=True"
  else PER_WORD=""
fi
echo "PER_WORD: ${PER_WORD}"

for file in /mnt/sdb_4/g2p/chpts/conformer/v${VERSION}/${JOB_ID}/g2p/G2PCTC/*/checkpoints/G2PCTC.nemo
do
  LOG_CLEAN=${JOB_ID}_log_clean.txt
  LOG_NO_CLEAN=${JOB_ID}_log_no_clean.txt


  # eval with --CLEAN
  output_heteronyms="CTC_${JOB_ID}.json"
  python g2p_ctc_inference.py pretrained_model=$file manifest_filepath=/mnt/sdb_4/g2p/data_ipa/evaluation_sets_v${VERSION}/CMU_TEST_MULTI/cmu_test.json batch_size=$BATCH_SIZE > "/tmp/cmu_clean.txt"
  python g2p_ctc_inference.py pretrained_model=$file manifest_filepath=/mnt/sdb_4/g2p/data_ipa/evaluation_sets_v${VERSION}/eval_wikihomograph.json output_file=$output_heteronyms batch_size=$BATCH_SIZE $PER_WORD > "/tmp/wiki_clean.txt"
  python evaluate_heteronyms.py --manifest=$output_heteronyms > "/tmp/wiki_heteronyms_clean.txt"
  tail -n 5 "/tmp/cmu_clean.txt" "/tmp/wiki_clean.txt" "/tmp/wiki_heteronyms_clean.txt" > ${LOG_CLEAN}

  output_heteronyms="CTC_${JOB_ID}.json"
  python g2p_ctc_inference.py pretrained_model=$file manifest_filepath=/mnt/sdb_4/g2p/data_ipa/evaluation_sets_v${VERSION}/CMU_TEST_MULTI/cmu_test.json batch_size=$BATCH_SIZE > "/tmp/cmu.txt"
  python g2p_ctc_inference.py pretrained_model=$file manifest_filepath=/mnt/sdb_4/g2p/data_ipa/evaluation_sets_v${VERSION}/eval_wikihomograph.json output_file=$output_heteronyms batch_size=$BATCH_SIZE > "/tmp/wiki.txt"
  python evaluate_heteronyms.py --manifest=$output_heteronyms > "/tmp/wiki_heteronyms.txt"
  tail -n 5 "/tmp/cmu.txt" "/tmp/wiki.txt" "/tmp/wiki_heteronyms.txt" > ${LOG_NO_CLEAN}

  echo "=======> Eval for ${JOB_ID} with CLEAN=True"
  cat ${LOG_CLEAN}
  echo "=======> Eval for ${JOB_ID} with CLEAN=False"
  cat ${LOG_NO_CLEAN}
done