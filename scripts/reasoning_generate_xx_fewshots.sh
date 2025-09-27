#!/bin/bash
#SBATCH --job-name=langb
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --partition=preempt
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --overcommit

# Example usage:
# for LANG in de fr es ru th te bn sw ja zh id
# for LANG in de es ja id
# for LANG in de fr es ru th te bn sw zh
# do
# PORT=$(( $RANDOM % (65535 - 1024 + 1) + 1024 ))
# sbatch lang_boot/scripts/reasoning_generate_xx_traces.sh \
#     Qwen/Qwen2.5-7B-Instruct \
#     gsm8k_train \
#     ${LANG} \
#     data/ \
#     ${PORT}
# done

. ./lang_boot/config/.env

while getopts ":m:l:n:t:d:s:f:r:v:g:e:j:p:w:x:" opt; do
  case ${opt} in
    m ) MODEL=$OPTARG;;
    l ) LANGUAGE=$OPTARG;;
    t ) TASK=$OPTARG;;
    d ) DATA_PATH=$OPTARG;;
    s ) SAVE_MODEL_PATH=$OPTARG;;
    f ) FUNCTION_NAME=$OPTARG;;
    r ) RUN_LABEL=$OPTARG;;
    v ) RUN_NUMBER=$OPTARG;;
    g ) USE_GCS=$OPTARG;;
    e ) SOURCE_TYPE=$OPTARG;;
    j ) USE_JUDGE=$OPTARG;;
    p ) USE_PRIVILEGED=$OPTARG;;
    w ) USE_REWARD_FN=$OPTARG;;
    x ) FULL_DATA_PATH=$OPTARG;;
    # \? ) echo "Usage: cmd [-u] [-p]";;
  esac
done


MODEL=$1
TASK=$2
LANG=$3
DATA_PATH=$4
PORT="${5:-8000}"
OTHER_ARGS=$6
PP_SIZE="${7:-1}"
TP_SIZE="${8:-1}"

MODEL_ALIAS=$(echo $MODEL | sed 's/\//-/g')


MAX_TOKEN=4096
vllm serve $MODEL \
    --port ${PORT} \
    --max_model_len ${MAX_TOKEN} \
    --pipeline_parallel_size ${PP_SIZE} \
    --tensor_parallel_size ${TP_SIZE} \
    --distributed-executor-backend mp > ${TMPDIR}vllm.txt &

    # --task "${TASK}t//${LANG}_generate_traces" \
    # --task local_json_taskt//${LANG}_translate \

yeval \
    --model $MODEL \
    --task ${TASK}_devt//${LANGUAGE}_reason \
    --include_path lang_boot/tasks/ \
    --api_base "http://localhost:${PORT}/v1" \
    --run_name $TASK+$LANGUAGE+generated+fewshot \
    --sample_args n=16,temperature=1.0,logprobs=True \
    --trust_remote_code \
    --output_path data/$MODEL_ALIAS/raw_traces/ $OTHER_ARGS
pkill vllm
sleep 2m
