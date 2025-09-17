#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --partition=preempt
#SBATCH --gres=gpu:L40S:4
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-node=1
#SBATCH --overcommit

# . ./lang_boot/config/.sft_env
. ./lang_boot/config/.env

# for LANGUAGE in id de es ja
# do
# sbatch lang_boot/scripts/train_sft.sh \
#     -m Qwen/Qwen2.5-7B \
#     -l ${LANGUAGE} \
#     -t deepscaler_train \
#     -f /data/user_data/lsutawik/lbr-language_bootstrap_reasoning/data/deepscaler_os/ \
#     -s /data/user_data/lsutawik/lbr-language_bootstrap_reasoning/ \
#     -r sft
# done

# sbatch lang_boot/scripts/train_sft.sh \
#     -m Qwen/Qwen2.5-7B \
#     -l en \
#     -t deepscaler_train \
#     -f /data/user_data/lsutawik/lbr-language_bootstrap_reasoning/data/deepscaler_os/ \
#     -s /data/user_data/lsutawik/lbr-language_bootstrap_reasoning/
#     -r sft

while getopts ":s:m:l:t:d:v:r:o:f:" opt; do
  case ${opt} in
    m ) MODEL=$OPTARG;;
    l ) LANGUAGE=$OPTARG;;
    t ) TASK=$OPTARG;;
    d ) DATA_PATH=$OPTARG;;
    v ) RUN_NUMBER=$OPTARG;;
    r ) RUN_LABEL=$OPTARG;;
    o ) OTHER_ARGS=$OPTARG;;
    s ) SAVE_MODEL_PATH=$OPTARG;;
    f ) FULL_DATA_PATH=$OPTARG;;
    # \? ) echo "Usage: cmd [-u] [-p]";;
  esac
done

MODEL_ALIAS=$(echo $MODEL | sed 's/\//-/g')
RUN_NUMBER="${RUN_NUMBER:-0}"
RUN_LABEL="${RUN_LABEL:-sft}"
RUN_NAME=${RUN_LABEL}+${MODEL_ALIAS}+${TASK}+${LANGUAGE}+${RUN_NUMBER}
DEFAULT_FULL_DATA_PATH=${DATA_PATH}prep_traces/${TASK}+${LANGUAGE}/
FULL_DATA_PATH="${FULL_DATA_PATH:-$DEFAULT_FULL_DATA_PATH}"
FULL_SAVE_PATH=${SAVE_MODEL_PATH}${RUN_NAME}
NUM_GPUS=$(nvidia-smi -L | wc -l)

echo $RUN_NAME
torchrun \
    --nproc-per-node=$NUM_GPUS \
    --rdzv-endpoint=0.0.0.0:29392 \
    -m verl.trainer.fsdp_sft_trainer \
        optim.lr=1e-5 \
        data.train_files=${FULL_DATA_PATH}train.parquet \
        data.val_files=${FULL_DATA_PATH}test.parquet \
        data.multiturn.enable=True \
        +data.messages_key=messages \
        +data.shuffle=True \
        +data.filter_overlong_prompts=True \
        data.truncation='left' \
        data.max_length=2048 \
        data.train_batch_size=32 \
        data.micro_batch_size_per_gpu=4 \
        model.partial_pretrain=${MODEL} \
        model.fsdp_config.offload_params=True \
        model.fsdp_config.model_dtype=bf16 \
        model.strategy=fsdp2 \
        trainer.default_local_dir=${FULL_SAVE_PATH}/checkpoints/ \
        trainer.project_name='lbr-lang_boot' \
        trainer.experiment_name=${RUN_NAME} \
        trainer.n_gpus_per_node=${NUM_GPUS} \
        trainer.save_freq=250 \
        trainer.total_epochs=10 \
        trainer.total_training_steps=1000 \
        trainer.logger=['console','wandb'] \
        trainer.checkpoint.save_contents=['hf_model']
        # model.use_liger=True \
        # data.prompt_key=input \
        # data.response_key=output \
        # trainer.logger="['console', 'wandb']"
        # data.prompt_key=extra_info \
        # data.response_key=extra_info \
        # data.prompt_dict_keys=['question'] \
        # +data.response_dict_keys=['answer'] \
        # +data.filter_overlong_prompts=True \

# for STEP in 500 450 400 350 300 250 200 150 100 50
# do
#     MODEL_STEP=global_step_${STEP}
#     PORT=$(( $RANDOM % (65535 - 1024 + 1) + 1024 ))
#     bash lang_boot/scripts/eval_mgsm.sh \
#         -s ${TMP_SAVE_PATH} \
#         -m ${RUN_NAME}/${MODEL_STEP} \
#         -l ${LANGUAGE} \
#         -r ${PORT} 
# done

# mv ${TMP_PATH} ${END_SAVE_PATH}
        # model.fsdp_config.model_dtype=bf16 \
        # model.fsdp_config.cpu_offload=True \
        # model.enable_gradient_checkpointing=True \
