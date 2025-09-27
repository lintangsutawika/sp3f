#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --partition=preempt
#SBATCH --gres=gpu:L40:1
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --overcommit

. ./lang_boot/config/.env

LANG="en"

# bash lang_boot/scripts/reasoning_construct_en_dataset.sh \
#     math_train \
#     data/Qwen-Qwen2.5-7B-Instruct/ \
#     -1

TASK=$1
DATA_PATH=$2
MAX_SAMPLES=$3

python lang_boot/lang_boot/construct.py \
    --response generated \
    --task ${TASK} \
    --lang ${LANG} \
    --use_lang \
    --use_accuracy \
    --max_samples ${MAX_SAMPLES} \
    --data_path ${DATA_PATH}
