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

while getopts ":s:m:r:o:p:t:e:" opt; do
  case ${opt} in
    s ) MODEL_PATH=$OPTARG;;
    m ) MODEL=$OPTARG;;
    r ) PORT=$OPTARG;;
    o ) OTHER_ARGS=$OPTARG;;
    p ) PP_SIZE=$OPTARG;;
    t ) TP_SIZE=$OPTARG;;
    e ) MODEL_SUFFIX=$OPTARG;;
    # \? ) echo "Usage: cmd [-p] [-m] [-l] [-o] [-pp] [-tp]";;
  esac
done

# sbatch lang_boot/scripts/eval_lang_id.sh \
#     -m r_acc-a-r_lang_fn-w-rmse+Qwen-Qwen2.5-7B+deepscaler_train+id+0 \
#     -s /data/user_data/lsutawik/lbr-language_bootstrap_reasoning/ \
#     -e /checkpoints/global_step_250/actor/huggingface/ -o "--overwrite"

RANDOM_PORT=$(( $RANDOM % (65535 - 1024 + 1) + 1024 ))
PORT="${PORT:-$(( $RANDOM_PORT ))}"
PP_SIZE="${PP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-1}"

TASK_LIST=(
    copal_id_colloquial_id
    copal_id_standard_id
    indo_mmlu_id
)

PROMPT_LANG_LIST=(
    id_system
    # id_measure
    # id_reason
    # en_reason
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

        # yeval \
        #     --model ${MODEL_PATH}${MODEL}${MODEL_SUFFIX} \
        #     --sample_args n=16,temperature=1.0,logprobs=True \
        #     --task "${TASK}t//${PROMPT}" \
        #     --include_path lang_boot/tasks/ \
        #     --api_base "http://localhost:${PORT}/v1" \
        #     --run_name $MODEL:$TASK:${PROMPT} \
        #     --trust_remote_code \
        #     --output_path ./data/coverage_scores/ $OTHER_ARGS
    done
done
