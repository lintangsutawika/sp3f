#!/bin/bash
#SBATCH --job-name=langb
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --partition=preempt
#SBATCH --gres=gpu:A6000:1
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --overcommit

. ./lang_boot/config/.sft_env
export VLLM_USE_V1=0

while getopts ":m:l:t:r:o:p:w:x:y:" opt; do
  case ${opt} in
    m ) MODEL=$OPTARG;;
    x ) MODEL_PATH=$OPTARG;;
    y ) DATA_PATH=$OPTARG;;
    l ) LANGUAGE=$OPTARG;;
    t ) TASK=$OPTARG;;
    r ) PORT=$OPTARG;;
    o ) OTHER_ARGS=$OPTARG;;
    p ) PP_SIZE=$OPTARG;;
    w ) TP_SIZE=$OPTARG;;
    # \? ) echo "Usage: cmd [-u] [-p]";;
  esac
done

# sbatch lang_boot/scripts/reasoning_translate_traces.sh \
#     -m Qwen/Qwen2.5-7B \
#     -l ${LANGUAGE} \
#     -t deepscaler_train \
#     -y /data/user_data/lsutawik/lbr-language_bootstrap_reasoning/data/Qwen-Qwen2.5-7B-deepscaler_en/ \
#     -x /data/user_data/lsutawik/lbr-language_bootstrap_reasoning/ -o "--n_samples 1000"

RANDOM_PORT=$(( $RANDOM % (65535 - 1024 + 1) + 1024 ))
PORT="${PORT:-$RANDOM_PORT}"
PP_SIZE="${PP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-1}"

MODEL_ALIAS=$(echo $MODEL | sed 's/\//-/g')
DATA_DIR="${DATA_PATH}raw_traces/${TASK}+en+generated+traces/output.jsonl"

MAX_TOKEN=8192
vllm serve $MODEL \
    --port ${PORT} \
    --max_model_len ${MAX_TOKEN} \
    --pipeline_parallel_size ${PP_SIZE} \
    --tensor_parallel_size ${TP_SIZE} \
    --distributed-executor-backend mp > ${TMPDIR}vllm.txt &

yeval \
    --model $MODEL \
    --task json_highest_log_${TASK}t//${LANGUAGE}_translate \
    --include_path lang_boot/tasks/ \
    --data_kwargs "{'data_files': '${DATA_DIR}'}" \
    --api_base "http://localhost:${PORT}/v1" \
    --run_name $TASK+$LANGUAGE+translated+traces \
    --sample_args n=4,temperature=1.0,logprobs=True \
    --trust_remote_code \
    --output_path ${MODEL_PATH}data/$MODEL_ALIAS/raw_traces/ $OTHER_ARGS

pkill vllm
sleep 2m
