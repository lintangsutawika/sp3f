#!/bin/bash
#SBATCH --job-name=langb
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:4
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --overcommit

# Example usage:
# sbatch lang_boot/scripts/reasoning_generate_en_traces.sh \
#     -m Qwen-Qwen2.5-7B-deepscaler_en \
#     -t deepscaler_train -w 8 -x /data/user_data/lsutawik/lbr-language_bootstrap_reasoning/

. ./lang_boot/config/.sft_env
export VLLM_USE_V1=0

while getopts ":m:t:r:o:p:w:x:y:" opt; do
  case ${opt} in
    m ) MODEL=$OPTARG;;
    x ) MODEL_PATH=$OPTARG;;
    y ) DATA_PATH=$OPTARG;;
    t ) TASK=$OPTARG;;
    r ) PORT=$OPTARG;;
    o ) OTHER_ARGS=$OPTARG;;
    p ) PP_SIZE=$OPTARG;;
    w ) TP_SIZE=$OPTARG;;
    # \? ) echo "Usage: cmd [-u] [-p]";;
  esac
done

RANDOM_PORT=$(( $RANDOM % (65535 - 1024 + 1) + 1024 ))
PORT="${PORT:-$RANDOM_PORT}"
PP_SIZE="${PP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-1}"

MODEL_ALIAS=$(echo $MODEL | sed 's/\//-/g')

MAX_TOKEN=4096
vllm serve ${MODEL_PATH}${MODEL} \
    --port ${PORT} \
    --max_model_len ${MAX_TOKEN} \
    --pipeline_parallel_size ${PP_SIZE} \
    --tensor_parallel_size ${TP_SIZE} \
    --distributed-executor-backend mp > ${TMPDIR}vllm.txt &

yeval \
    --model ${MODEL_PATH}$MODEL \
    --task "${TASK}t//en_system" \
    --include_path lang_boot/tasks/ \
    --api_base "http://localhost:${PORT}/v1" \
    --run_name $TASK+en+generated+traces \
    --sample_args n=16,temperature=1.0,logprobs=True \
    --trust_remote_code \
    --output_path ${DATA_PATH}data/$MODEL_ALIAS/raw_traces/ $OTHER_ARGS
pkill vllm
sleep 2m
