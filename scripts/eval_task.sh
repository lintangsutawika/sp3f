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

# -m simplescaling/s1.1-7B
# -m Qwen/Qwen2.5-7B
# -m Qwen/Qwen2.5-7B-Instruct
# for LANGUAGE in id ja es sw bn te
# do
#     sbatch lang_boot/scripts/eval_task.sh \
#         -m Qwen/Qwen2.5-7B \
#         -l ${LANGUAGE} -o "--overwrite"
# done

# sbatch lang_boot/scripts/eval_task.sh \
#     -m {TRAINED_MODEL} \
#     -s /data/user_data/lsutawik/lbr-language_bootstrap_reasoning/ \
#     -e /checkpoints/global_step_200/actor/huggingface/ \
#     -l id -o "--overwrite"

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
    ${LANGUAGE}_system
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
        yeval \
            --model ${MODEL_PATH}${MODEL}${MODEL_SUFFIX} \
            --sample_args "n=8,temperature=0.7,logprobs=True,top_p=0.8" \
            --task "${TASK}t//${PROMPT}" \
            --include_path lang_boot/tasks/ \
            --api_base "http://localhost:${PORT}/v1" \
            --run_name $MODEL+$TASK+${PROMPT} \
            --trust_remote_code \
            --output_path ./data/eval_scores/ $OTHER_ARGS

    done
done
pkill vllm
sleep 2m
