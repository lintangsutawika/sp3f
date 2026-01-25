#!/bin/bash

# Environment Variables
. .env

while getopts ":s:m:l:t:d:v:r:o:f:" opt; do
  case ${opt} in
    m ) MODEL=$OPTARG;;
    l ) LANGUAGE=$OPTARG;;
    t ) TASK=$OPTARG;;
    d ) DATA_PATH=$OPTARG;;
    v ) RUN_NUMBER=$OPTARG;;
    r ) RUN_LABEL=$OPTARG;;
    o ) OTHER_ARGS=$OPTARG;;
    s ) SAVE_MODEL_PATH=$OPTARG;;
    f ) FULL_DATA_PATH=$OPTARG;;
    \? ) echo "Usage: cmd [-u] [-p]";;
  esac
done

MODEL_ALIAS=$(echo $MODEL | sed 's/\//-/g')
RUN_NUMBER="${RUN_NUMBER:-0}"
RUN_LABEL="${RUN_LABEL:-sft}"
RUN_NAME=${RUN_LABEL}+${MODEL_ALIAS}+${TASK}+${LANGUAGE}+${RUN_NUMBER}
DEFAULT_FULL_DATA_PATH=${DATA_PATH}prep_traces/${TASK}+${LANGUAGE}/
FULL_DATA_PATH="${FULL_DATA_PATH:-$DEFAULT_FULL_DATA_PATH}"
FULL_SAVE_PATH=${SAVE_MODEL_PATH}${RUN_NAME}
NUM_GPUS=$(nvidia-smi -L | wc -l)

echo $RUN_NAME
torchrun \
    --nproc-per-node=$NUM_GPUS \
    --rdzv-endpoint=0.0.0.0:29392 \
    -m verl.trainer.fsdp_sft_trainer \
        optim.lr=1e-4 \
        data.train_files=${FULL_DATA_PATH}train.parquet \
        data.val_files=${FULL_DATA_PATH}test.parquet \
        data.multiturn.enable=True \
        +data.messages_key=messages \
        +data.shuffle=True \
        +data.filter_overlong_prompts=True \
        data.truncation='left' \
        data.max_length=4192 \
        data.train_batch_size=16 \
        data.micro_batch_size_per_gpu=1 \
        model.partial_pretrain=${MODEL} \
        model.fsdp_config.offload_params=True \
        model.fsdp_config.model_dtype=bf16 \
        model.strategy=fsdp2 \
        trainer.default_local_dir=${FULL_SAVE_PATH}/checkpoints/ \
        trainer.project_name='sp3f' \
        trainer.experiment_name=${RUN_NAME} \
        trainer.n_gpus_per_node=${NUM_GPUS} \
        trainer.save_freq=500 \
        trainer.total_epochs=1 \
        trainer.total_training_steps=1000 \
        trainer.logger=['console','wandb'] \
        trainer.checkpoint.save_contents=['hf_model']
