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

. ./lang_boot/config/.sft_env
export VLLM_USE_V1=0

while getopts ":m:l:t:r:o:p:w:x:y:z:" opt; do
  case ${opt} in
    m ) MODEL=$OPTARG;;
    x ) MODEL_PATH=$OPTARG;;
    z ) MODEL_SUFFIX=$OPTARG;;
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

# sbatch lang_boot/scripts/reasoning_generate_xx_traces.sh \
#     -m r_acc-a-r_lang_fn-w-rmse+Qwen-Qwen2.5-7B+deepscaler_train+${LANGUAGE}+0 \
#     -l ${LANGUAGE} \
#     -t deepscaler_train \
#     -y /data/user_data/lsutawik/lbr-language_bootstrap_reasoning/data/Qwen-Qwen2.5-7B/ \
#     -x /data/user_data/lsutawik/lbr-language_bootstrap_reasoning/ \
#     -z /checkpoints/global_step_250/actor/huggingface/ -o "--n_samples 5000 --overwrite"

RANDOM_PORT=$(( $RANDOM % (65535 - 1024 + 1) + 1024 ))
PORT="${PORT:-$RANDOM_PORT}"
PP_SIZE="${PP_SIZE:-1}"
TP_SIZE="${TP_SIZE:-1}"


MODEL_ALIAS=$(echo $MODEL | sed 's/\//-/g')
DATA_DIR="${DATA_PATH}raw_traces/${TASK}+${LANGUAGE}+translated+queries/output.jsonl"


MAX_TOKEN=4096
vllm serve ${MODEL_PATH}$MODEL${MODEL_SUFFIX} \
    --port ${PORT} \
    --max_model_len ${MAX_TOKEN} \
    --pipeline_parallel_size ${PP_SIZE} \
    --tensor_parallel_size ${TP_SIZE} \
    --distributed-executor-backend mp > ${TMPDIR}vllm.txt &

    # --task "${TASK}t//${LANG}_generate_traces" \
    # --task local_json_taskt//${LANG}_translate \

yeval \
    --model ${MODEL_PATH}$MODEL${MODEL_SUFFIX} \
    --task json_highest_lang_${TASK}t//${LANGUAGE}_system \
    --include_path lang_boot/tasks/ \
    --data_kwargs "{'data_files': '${DATA_DIR}'}" \
    --api_base "http://localhost:${PORT}/v1" \
    --run_name $TASK+$LANGUAGE+generated+traces \
    --sample_args n=16,temperature=1.0,logprobs=True \
    --trust_remote_code \
    --output_path ${MODEL_PATH}data/$MODEL_ALIAS/raw_traces/ $OTHER_ARGS
pkill vllm
sleep 2m
