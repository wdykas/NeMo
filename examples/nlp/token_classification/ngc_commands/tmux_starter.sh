script_name="$1"
dataset_id="$2"
WANDB_API_KEY="$3"
exp_name="$4"
if [ -z "$5" ]; then
  gpus=8
else
  gpus="$5"
fi
if [ -z "$6" ]; then
  gpu_memory=16
else
  gpu_memory="$6"
fi

read -r -d '' command << EOF
set -x -e
tmux new-session -d -s "work" 'bash' 2>&1 | tee -a /result/tmux_logs.txt
ls /workspace/NeMo/examples/nlp/token_classification/ngc_commands/ 2>&1 | tee -a /result/tmux_logs.txt
tmux new-window -n run -t work: 'bash /workspace/NeMo/examples/nlp/token_classification/ngc_commands/${script_name} \
  ${WANDB_API_KEY} \
  ${exp_name} \
  ${gpus}' 2>&1 | tee -a /result/tmux_logs.txt
sleep 1000000
set +x +e
EOF

ngc batch run \
  --instance "dgx1v.${gpu_memory}g.${gpus}.norm" \
  --name "ml-model.bert ${exp_name}" \
  --image "nvcr.io/nvidian/ac-aiapps/punctuation-and-capitalization:latest" \
  --result /result \
  --datasetid ${dataset_id}:/data \
  --commandline "${command}"