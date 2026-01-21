#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --partition=general
#SBATCH --gres=gpu:A6000:1
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=512G
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --overcommit

. ./lang_boot/config/.env

while getopts ":s:m:l:r:o:p:t:e:d:" opt; do
  case ${opt} in
    s ) MODEL_PATH=$OPTARG;;
    m ) MODEL=$OPTARG;;
    l ) LANGUAGE=$OPTARG;;
    r ) PORT=$OPTARG;;
    o ) OTHER_ARGS=$OPTARG;;
    p ) PP_SIZE=$OPTARG;;
    t ) TP_SIZE=$OPTARG;;
    e ) MODEL_SUFFIX=$OPTARG;;
    d ) LOCAL_DATA_PATH=$OPTARG;;
    # \? ) echo "Usage: cmd [-p] [-m] [-l] [-o] [-pp] [-tp]";;
  esac
done

# for LANGUAGE in id ja es sw bn
# do
#     sbatch lang_boot/scripts/eval_task_tt_02_inference_en.sh \
#         -m Qwen/Qwen2.5-7B-Instruct \
#         -l ${LANGUAGE} -o "--overwrite"
# done

RANDOM_PORT=$(( $RANDOM % (65535 - 1024 + 1) + 1024 ))
PORT="${PORT:-$(( $RANDOM_PORT ))}"
PP_SIZE="${PP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-1}"
MODEL_ALIAS=$(echo $MODEL | sed 's/\//-/g')
DEFAULT_DATA_PATH="data/$MODEL_ALIAS/"
LOCAL_DATA_PATH="${LOCAL_DATA_PATH:-$DEFAULT_DATA_PATH}"

echo $LOCAL_DATA_PATH

TASK_LIST=(
    # mgsm_
    # global_mmlu_
    # belebele_
    mt_math100_
)

PROMPT_LANG_LIST=(
    en_system
)

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
        SHOW_TASK_NAME="${TASK}${LANGUAGE}"
        DATA_DIR="${LOCAL_DATA_PATH}en_queries/${SHOW_TASK_NAME}+en/output.jsonl"
        FULL_TASK_NAME="${TASK}translate"
        yeval \
            --model ${MODEL_PATH}${MODEL}${MODEL_SUFFIX} \
            --sample_args "n=8,temperature=0.7,logprobs=True,top_p=0.8" \
            --task "${FULL_TASK_NAME}t//${PROMPT}" \
            --data_kwargs "{'data_files': '${DATA_DIR}'}" \
            --include_path lang_boot/tasks/ \
            --api_base "http://localhost:${PORT}/v1" \
            --run_name ${MODEL}_Translate_Test+$SHOW_TASK_NAME+${LANGUAGE}_system \
            --trust_remote_code \
            --output_path ./data/$MODEL_ALIAS/en_traces/ $OTHER_ARGS


        # yeval \
        #     --model ${MODEL_PATH}${MODEL}${MODEL_SUFFIX} \
        #     --sample_args n=16,temperature=1.0,logprobs=True \
        #     --task "${TASK}t//${PROMPT}" \
        #     --include_path lang_boot/tasks/ \
        #     --api_base "http://localhost:${PORT}/v1" \
        #     --run_name $MODEL+$TASK+${PROMPT} \
        #     --trust_remote_code \
        #     --output_path ./data/coverage_scores/ $OTHER_ARGS
    done
done
# pkill vllm
# sleep 2m
