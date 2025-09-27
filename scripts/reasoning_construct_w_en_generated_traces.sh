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

# LANG="de"
# LANG="fr"
# LANG="es"
# LANG="ru"
# LANG="zh"
# LANG="ja"
# LANG="th"
# LANG="te"
# LANG="bn"
# LANG="sw"

# for LANG in de es ja id
# for LANG in id
# do
# bash lang_boot/scripts/reasoning_construct_w_en_generated_traces.sh \
#     math_train \
#     ${LANG} \
#     data/Qwen-Qwen2.5-7B-Instruct/ \
#     -1
# done

TASK=$1
LANG=$2
DATA_PATH=$3
MAX_SAMPLES=$4

python lang_boot/lang_boot/construct.py \
    --response generated \
    --task ${TASK} \
    --lang ${LANG} \
    --use_en \
    --use_lang \
    --use_accuracy \
    --max_samples ${MAX_SAMPLES} \
    --data_path ${DATA_PATH}
