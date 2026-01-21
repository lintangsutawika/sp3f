#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1

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
    mt_math100_$LANGUAGE
    belebele_$LANGUAGE
    global_mmlu_$LANGUAGE
    # global_piqa_$LANGUAGE
)

PROMPT_LANG_LIST=(
    ${LANGUAGE}_system
)

MAX_TOKEN=8192
# Detect number of GPUs
# NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
#     --data-parallel-size ${NUM_GPUS} \
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
        # --sample_args "n=8,temperature=0.7,logprobs=True,top_p=0.8,extra_body={'chat_template_kwargs':{'enable_thinking':False}}" \
        yeval \
            --model ${MODEL_PATH}${MODEL}${MODEL_SUFFIX} \
            --sample_args "n=8,temperature=0.7,logprobs=True,top_p=0.8" \
            --task "${TASK}t//${PROMPT}" \
            --include_path lang_boot/tasks/ \
            --api_base "http://localhost:${PORT}/v1" \
            --run_name $MODEL+$TASK+${PROMPT} \
            --trust_remote_code \
            --output_path /data/user_data/lsutawik/lbr-language_bootstrap_reasoning/data/eval_scores/ $OTHER_ARGS
            # --run_name $MODEL-thinking+$TASK+${PROMPT} \
    done
done
#pkill vllm
#sleep 2m
