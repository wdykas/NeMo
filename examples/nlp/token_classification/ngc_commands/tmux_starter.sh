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
tmux new-session -d -s "work" 'bash'
ls /workspace/NeMo/examples/tts/commands/ >> logs.txt
tmux new-window -n run -t work: 'bash /workspace/NeMo/examples/tts/commands/${script_name} ${WANDB_API_KEY} ${exp_name} ${gpus}'
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