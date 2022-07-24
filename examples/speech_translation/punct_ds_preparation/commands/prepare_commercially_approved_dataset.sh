set -e -x

modes="${@:-all}"
echo "Running in modes: ${modes}"
sleep 10
modes=( ${modes} )

DATA_DIR=/media/apeganov/DATA/punctuation_and_capitalization/simplest/3_128
RESULT_DATASET_NAME=commercial

dataset_names=(
  pubmed_x1_17.07.2022
  pg19_x1_19.07.2022
  un_x1_11.06.2022
  google_normalization_x1_03.05.2022
  europarl_raw_x1_05.06.2022
  tatoeba_x1_03.05.2022
)

num_dev_lines=(2000 2000 5000 12000 1000 15000)
num_test_lines=(4000 4000 10000 24000 2000 30000)


for i in "${!dataset_names[@]}"; do
  dataset_dir="${DATA_DIR}/${dataset_names[$i]}"
  echo "Processing ${dataset_dir}"
  if [[ " ${modes[*]} " == " unite " || " ${modes[*]} " == " all " ]]; then
    python unite_documents_into_plain_text.py \
      --input_dir "${dataset_dir}/documents" \
      --output "${dataset_dir}/united_documents.txt"
  fi
  remaining_file="${dataset_dir}/remaining.txt"
  if [[ " ${modes[*]} " == " extract " || " ${modes[*]} " == " all " ]]; then
    python extract_lines.py \
      --input_file "${dataset_dir}/united_documents.txt" \
      --extracted_file "${dataset_dir}/dev_lines.txt" \
      --remaining_file "${remaining_file}" \
      --num_lines_to_extract "${num_dev_lines[$i]}"
  fi
  if [[ " ${modes[*]} " == " extract " || " ${modes[*]} " == " all " ]]; then
    python extract_lines.py \
      --input_file "${remaining_file}" \
      --extracted_file "${dataset_dir}/test_lines.txt" \
      --remaining_file "${dataset_dir}/train_lines.txt" \
      --num_lines_to_extract "${num_test_lines[$i]}"
  fi
  for fn in dev_lines.txt test_lines.txt train_lines.txt; do
    if [[ " ${modes[*]} " == " sentences " || " ${modes[*]} " == " all " ]]; then
      python join_sentences_with_ratios.py \
        --input_file "${dataset_dir}/${fn}" \
        --output_file "${dataset_dir}/${fn%%_lines.txt}_sentences.txt"
    fi
    if [[ " ${modes[*]} " == " segments " || " ${modes[*]} " == " all " ]]; then
      python cut_segments.py \
        --input_file "${dataset_dir}/${fn}" \
        --output_file "${dataset_dir}/${fn%%_lines.txt}_segments.txt"
    fi
  done
  if [[ " ${modes[*]} " == " labels " || " ${modes[*]} " == " all " ]]; then
    for fn in dev_segments.txt dev_sentences.txt test_segments.txt test_sentences.txt train_segments.txt train_sentences.txt; do
      python text_to_punc_cap_dataset.py \
        --input_text "${dataset_dir}/${fn}" \
        --output_dir "${dataset_dir}/${fn%%.txt}" \
        --create_model_input \
        --bert_labels \
        --allowed_punctuation ',.?' \
        --num_jobs 1
    done
  fi
done

output_dataset_dir="${DATA_DIR}/${RESULT_DATASET_NAME}"
not_shuffled_dir="${output_dataset_dir}/train_not_shuffled"
if [[ " ${modes[*]} " == " cat " || " ${modes[*]} " == " all " ]]; then
  mkdir -p "${not_shuffled_dir}"
  for fn in input.txt bert_labels.txt text.txt; do
    not_shuffled_file="${not_shuffled_dir}/${fn}"
    > "${not_shuffled_file}"
    for ds_name in "${dataset_names[@]}"; do
      cat "${dataset_dir}/${ds_name}/train_"{segments,sentences}"/${fn}" >> "${not_shuffled_file}"
    done
  done
fi

shuffled_dir="${output_dataset_dir}/train"
if [[ " ${modes[*]} " == " shuffle " || " ${modes[*]} " == " all " ]]; then
  python shuffle_jointly.py \
    --input_files "${not_shuffled_dir}/"{input,bert_labels,text}".txt" \
    --output_files "${shuffled_dir}/"{input,bert_labels,text}".txt" \
    --line_delimiter $'\t' \
    --tmp_dir "${output_dataset_dir}/tmp"
fi

