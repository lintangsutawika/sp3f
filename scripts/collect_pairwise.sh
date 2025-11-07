#!/bin/bash
#SBATCH --job-name=grpo
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:8
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=512G
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=babel-9-3,babel-4-25,babel-14-29,babel-12-9,babel-13-1,babel-7-1,babel-13-13,babel-10-9,babel-7-9,babel-2-13

. ./lang_boot/config/.env

while getopts ":l:" opt; do
  case ${opt} in
    l ) LANGUAGE=$OPTARG;;
    \? ) echo "Usage: cmd [-u] [-p]";;
  esac
done

uv run lang_boot/lang_boot/judge_responses.py \
    --language ${LANGUAGE} \
    --model_a r_privileged_API_GPT_4o_Mini-r_acc-r_threshold-r_parsable+sft-Qwen2.5-7B-deepscaler_train+deepscaler_train+LANGUAGE+0 \
    --model_b Qwen/Qwen2.5-7B-Instruct

uv run lang_boot/lang_boot/judge_responses.py \
    --language ${LANGUAGE} \
    --model_a r_acc-r_threshold-r_parsable+sft-Qwen2.5-7B-deepscaler_train+deepscaler_train+LANGUAGE+0 \
    --model_b Qwen/Qwen2.5-7B-Instruct

uv run lang_boot/lang_boot/judge_responses.py \
    --language ${LANGUAGE} \
    --model_a r_privileged_API_GPT_4o_Mini-r_acc-r_threshold-r_parsable+sft-Qwen2.5-7B-deepscaler_train+deepscaler_train+LANGUAGE+0 \
    --model_b r_acc-r_threshold-r_parsable+sft-Qwen2.5-7B-deepscaler_train+deepscaler_train+LANGUAGE+0

# uv run lang_boot/lang_boot/judge_responses.py \
#     --language ${LANGUAGE} \
#     --model_a r_privileged_API_GPT_4o_Mini-r_acc-r_threshold-r_parsable+sft-Qwen2.5-7B-deepscaler_train+deepscaler_train+LANGUAGE+0 \
#     --model_b r_nonprivileged_API_GPT_4o_Mini-r_acc-r_threshold-r_parsable+sft-Qwen2.5-7B-deepscaler_train+deepscaler_train+LANGUAGE+0

# uv run lang_boot/lang_boot/judge_responses.py \
#     --language ${LANGUAGE} \
#     --model_a r_nonprivileged_API_GPT_4o_Mini-r_acc-r_threshold-r_parsable+sft-Qwen2.5-7B-deepscaler_train+deepscaler_train+LANGUAGE+0 \
#     --model_b Qwen/Qwen2.5-7B-Instruct

# uv run lang_boot/lang_boot/judge_responses.py \
#     --language ${LANGUAGE} \
#     --model_a r_nonprivileged_API_GPT_4o_Mini-r_acc-r_threshold-r_parsable+sft-Qwen2.5-7B-deepscaler_train+deepscaler_train+LANGUAGE+0 \
#     --model_b r_acc-r_threshold-r_parsable+sft-Qwen2.5-7B-deepscaler_train+deepscaler_train+LANGUAGE+0
