#!/bin/bash
#SBATCH --job-name=eval
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

. ./lang_boot/config/.env

while getopts ":s:m:l:r:o:p:t:e:" opt; do
  case ${opt} in
    s ) MODEL_PATH=$OPTARG;;
    m ) MODEL=$OPTARG;;
    l ) LANGUAGE=$OPTARG;;
    r ) PORT=$OPTARG;;
    o ) OTHER_ARGS=$OPTARG;;
    p ) PP_SIZE=$OPTARG;;
    t ) TP_SIZE=$OPTARG;;
    e ) MODEL_SUFFIX=$OPTARG;;
    # \? ) echo "Usage: cmd [-p] [-m] [-l] [-o] [-pp] [-tp]";;
  esac
done

# for LANGUAGE in id ja es sw
# do
#     sbatch lang_boot/scripts/eval_translate.sh \
#         -m Qwen/Qwen3-4B \
#         -l ${LANGUAGE}
# done

RANDOM_PORT=$(( $RANDOM % (65535 - 1024 + 1) + 1024 ))
PORT="${PORT:-$(( $RANDOM_PORT ))}"
PP_SIZE="${PP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-1}"

TASK_LIST=(
    mgsm_$LANGUAGE
    global_mmlu_$LANGUAGE
    belebele_$LANGUAGE
    mt_math100_$LANGUAGE
)

PROMPT_LANG_LIST=(
    en_translate
)

MODEL_ALIAS=$(echo $MODEL | sed 's/\//-/g')

MAX_TOKEN=8192
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
            --sample_args "n=4,temperature=0.7,logprobs=True,top_p=0.8,extra_body={'top_k': 20}" \
            --task "${TASK}t//${PROMPT}" \
            --include_path lang_boot/tasks/ \
            --api_base "http://localhost:${PORT}/v1" \
            --run_name $TASK+en \
            --trust_remote_code \
            --output_path ./data/$MODEL_ALIAS/eval_traces/ $OTHER_ARGS

    done
done
pkill vllm
# sleep 2m
