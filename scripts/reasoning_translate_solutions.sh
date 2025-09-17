#!/bin/bash
#SBATCH --job-name=langb
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --partition=preempt
#SBATCH --gres=gpu:L40S:4
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --overcommit

# Example usage:
# for LANG in de fr es ru th te bn sw ja zh id
# for LANG in de fr es ru th te bn sw zh
# for LANG in ja id
# do

# sbatch lang_boot/scripts/reasoning_translate_solutions.sh \
#     -m Qwen/Qwen2.5-7B \
#     -l id \
#     -t deepscaler_train -w 4
# done

. ./lang_boot/config/.sft_env
export VLLM_USE_V1=0

while getopts ":m:l:t:r:o:p:w:" opt; do
  case ${opt} in
    m ) MODEL=$OPTARG;;
    l ) LANGUAGE=$OPTARG;;
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

MAX_TOKEN=2048
vllm serve $MODEL \
    --port ${PORT} \
    --max_model_len ${MAX_TOKEN} \
    --pipeline_parallel_size ${PP_SIZE} \
    --tensor_parallel_size ${TP_SIZE} \
    --distributed-executor-backend mp > ${TMPDIR}vllm.txt &

yeval \
    --model $MODEL \
    --task ${TASK}_solutiont//${LANGUAGE}_translate \
    --include_path lang_boot/tasks/ \
    --api_base "http://localhost:${PORT}/v1" \
    --run_name $TASK+$LANGUAGE+translated+solutions \
    --sample_args n=16,temperature=1.0,logprobs=True \
    --trust_remote_code \
    --output_path data/$MODEL_ALIAS/raw_traces/ $OTHER_ARGS

pkill vllm
sleep 2m
