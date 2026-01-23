#!/bin/bash

# Environment Variables
. .env

while getopts ":a:s:m:l:r:o:p:t:e:d:" opt; do
  case ${opt} in
    a ) MODEL_ALIAS=$OPTARG;;
    s ) MODEL_PATH=$OPTARG;;
    m ) MODEL=$OPTARG;;
    l ) LANGUAGE=$OPTARG;;
    r ) PORT=$OPTARG;;
    o ) OTHER_ARGS=$OPTARG;;
    p ) PP_SIZE=$OPTARG;;
    t ) TP_SIZE=$OPTARG;;
    e ) MODEL_SUFFIX=$OPTARG;;
    d ) SAVE_PATH=$OPTARG;;
    \? ) echo "Usage: cmd [-u] [-p]";;
  esac
done

DEFAULT_MODEL_ALIAS=$(echo $MODEL | sed 's/\//-/g')
MODEL_ALIAS="${MODEL_ALIAS:-$DEFAULT_MODEL_ALIAS}"
RANDOM_PORT=$(( $RANDOM % (65535 - 1024 + 1) + 1024 ))
PORT="${PORT:-$(( $RANDOM_PORT ))}"
PP_SIZE="${PP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-1}"
SAVE_PATH="${SAVE_PATH:-./outputs/}"

TASK_LIST=(
    mgsm_$LANGUAGE
    mt_math100_$LANGUAGE
    belebele_$LANGUAGE
    global_mmlu_$LANGUAGE
)

PROMPT_LANG_LIST=(
    ${LANGUAGE}_system
)

MAX_TOKEN=2048
vllm serve ${MODEL_PATH}${MODEL}${MODEL_SUFFIX} \
    --port ${PORT} \
    --max_model_len ${MAX_TOKEN} \
    --pipeline_parallel_size ${PP_SIZE} \
    --tensor_parallel_size ${TP_SIZE} \
    --distributed-executor-backend mp > ${TMPDIR}vllm.txt &

for TASK in ${TASK_LIST[@]}
do
    for PROMPT in ${PROMPT_LANG_LIST[@]}
    do
        yeval \
            --model ${MODEL_PATH}${MODEL}${MODEL_SUFFIX} \
            --sample_args "n=8,temperature=0.7,logprobs=True,top_p=0.8" \
            --task "${TASK}t//${PROMPT}" \
            --include_path lang_boot/tasks/ \
            --api_base "http://localhost:${PORT}/v1" \
            --run_name $MODEL_ALIAS+$TASK+${PROMPT} \
            --trust_remote_code \
            --output_path ${SAVE_PATH}data/eval_scores/ $OTHER_ARGS
    done
done
