#!/bin/bash



JOB_ID=$1
BATCH_SIZE=64
PER_WORD=${2:-"None"}
VERSION=8

if [[ ${PER_WORD,,} == "true" ]]; then
  PER_WORD="--per_word "
  else PER_WORD=""
fi
echo "PER_WORD: ${PER_WORD}"

for file in /mnt/sdb_4/g2p/chpts/t5_generative/v${VERSION}/${JOB_ID}/g2p/T5G2P/*/checkpoints/T5G2P.nemo
do
  LOG_CLEAN=${JOB_ID}_log_clean.txt
  LOG_NO_CLEAN=${JOB_ID}_log_no_clean.txt


#  # eval with --CLEAN
#  output_heteronyms="T5_generative_${JOB_ID}.json"
#  python ../evaluate_t5_g2p.py --model_ckpt=$file --manifest_filepath=/mnt/sdb_4/g2p/data_ipa/evaluation_sets_v${VERSION}/CMU_TEST_MULTI/cmu_test.json --batch_size=$BATCH_SIZE --clean > "/tmp/cmu_clean.txt"
#  python ../evaluate_t5_g2p.py --model_ckpt=$file --manifest_filepath=/mnt/sdb_4/g2p/data_ipa/evaluation_sets_v${VERSION}/eval_wikihomograph.json --output=$output_heteronyms --batch_size=$BATCH_SIZE --clean ${PER_WORD} > "/tmp/wiki_clean.txt"
#  python evaluate_heteronyms.py --manifest_$output_heteronyms > "/tmp/wiki_heteronyms_clean.txt"
#  tail -n 5 "/tmp/cmu_clean.txt" "/tmp/wiki_clean.txt" "/tmp/wiki_heteronyms_clean.txt" > ${LOG_CLEAN}

  output_heteronyms="T5_generative_${JOB_ID}.json"
  python ../evaluate_t5_g2p.py --model_ckpt=$file --manifest_filepath=/mnt/sdb_4/g2p/data_ipa/evaluation_sets_v${VERSION}/CMU_TEST_MULTI/cmu_test.json --batch_size=$BATCH_SIZE > "/tmp/cmu.txt"
  python ../evaluate_t5_g2p.py --model_ckpt=$file --manifest_filepath=/mnt/sdb_4/g2p/data_ipa/evaluation_sets_v${VERSION}/eval_wikihomograph.json --output=$output_heteronyms --batch_size=$BATCH_SIZE ${PER_WORD}> "/tmp/wiki.txt"
  python evaluate_heteronyms.py --manifest=$output_heteronyms > "/tmp/wiki_heteronyms.txt"
  tail -n 5 "/tmp/cmu.txt" "/tmp/wiki.txt" "/tmp/wiki_heteronyms.txt" > ${LOG_NO_CLEAN}

  echo "=======> Eval for ${JOB_ID} with CLEAN=True"
  cat ${LOG_CLEAN}
  echo "=======> Eval for ${JOB_ID} with CLEAN=False"
  cat ${LOG_NO_CLEAN}
  echo ${file}
done