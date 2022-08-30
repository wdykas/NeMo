TASKNAME=$1
MODEL_SIZE=$2
NUM_VIRTUAL_TOKENS=$3
EPOCHS=$4
BATCH_SIZE=$5
LR=$6

NUM_DEVICES=$(nvidia-smi  -L | wc -l)
NUM_NODES=1

if [[ $MODEL_SIZE == "126m" ]]; then
    TENSOR_PARALLEL_SIZE=1
    PIPELINE_PARALLEL_SIZE=1
elif [[ $MODEL_SIZE == "1.3b" ]]; then
    TENSOR_PARALLEL_SIZE=1
    PIPELINE_PARALLEL_SIZE=1
elif [[ $MODEL_SIZE == "5b" ]]; then
    TENSOR_PARALLEL_SIZE=2
    PIPELINE_PARALLEL_SIZE=1
elif [[ $MODEL_SIZE == "20b" ]]; then
    TENSOR_PARALLEL_SIZE=8
    PIPELINE_PARALLEL_SIZE=1
elif [[ $MODEL_SIZE == "40b" ]]; then
    TENSOR_PARALLEL_SIZE=8
    PIPELINE_PARALLEL_SIZE=4
else
    echo "Language model size not recognized"
    exit 1
fi
    
TRAIN_FILE="train_data.jsonl"
VAL_FILE="val_data.jsonl"

RUN_NAME="${TASKNAME}_${MODEL_SIZE}_${NUM_VIRTUAL_TOKENS}_${EPOCHS}_${LR}"


echo "----Starting training----" \
&& python /workspace/NeMo/examples/nlp/language_modeling/megatron_gpt_prompt_learning.py \
        --config-name=megatron_api_service_prompt_learning.yaml \
        name=${RUN_NAME} \
        tokens=${NUM_VIRTUAL_TOKENS} \
        taskname=${TASKNAME} \
        trainer.devices=${NUM_DEVICES} \
        trainer.num_nodes=${NUM_NODES} \
        trainer.max_epochs=${EPOCHS} \
        trainer.val_check_interval=1.0 \
        exp_manager.exp_dir=/workspace \
        exp_manager.resume_ignore_no_checkpoint=True \
        model.language_model_path=/workspace/gpt_model.nemo \
        model.nemo_path=/results/${RUN_NAME}.nemo \
        model.existing_tasks=[] \
        model.new_tasks=[${TASKNAME}] \
        model.tensor_model_parallel_size=${TENSOR_PARALLEL_SIZE} \
        model.pipeline_model_parallel_size=${PIPELINE_PARALLEL_SIZE} \
        model.global_batch_size=${BATCH_SIZE} \
        model.micro_batch_size=2 \
        model.optim.lr=${LR} \
        model.data.train_ds=[/workspace/data/${TRAIN_FILE}] \
        model.data.validation_ds=[/workspace/data/${VAL_FILE}] \
        model.virtual_prompt_style=p-tuning \
        model.p_tuning.num_layers=1

