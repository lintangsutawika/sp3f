#!/bin/bash

# Environment Variables
. .env

while getopts ":a:m:l:n:t:d:s:f:u:r:v:g:e:j:p:w:y:o:b:z:i:k:c:" opt; do
  case ${opt} in
    a ) MODEL_ALIAS=$OPTARG;;
    m ) MODEL=$OPTARG;;
    l ) LANGUAGE=$OPTARG;;
    t ) TASK=$OPTARG;;
    d ) DATA_PATH=$OPTARG;;
    n ) N_ROLLOUTS=$OPTARG;;
    s ) SAVE_MODEL_PATH=$OPTARG;;
    f ) FUNCTION_NAME=$OPTARG;;
    u ) FUNCTION_PATH=$OPTARG;;
    r ) RUN_LABEL=$OPTARG;;
    v ) RUN_NUMBER=$OPTARG;;
    g ) USE_GCS=$OPTARG;;
    e ) SOURCE_TYPE=$OPTARG;;
    j ) USE_JUDGE=$OPTARG;;
    c ) USE_JUDGE_ALT=$OPTARG;;
    p ) USE_PRIVILEGED=$OPTARG;;
    w ) USE_REWARD_FN=$OPTARG;;
    y ) MODEL_PATH=$OPTARG;;
    o ) OTHER_ARGS=$OPTARG;;
    b ) DEBUG=$OPTARG;;
    z ) FULL_DATA_PATH=$OPTARG;;
    i ) USE_API=$OPTARG;;
    k ) JUDGE=$OPTARG;;
    \? ) echo "Usage: cmd [-u] [-p]";;
  esac
done

DEFAULT_MODEL_ALIAS=$(echo $MODEL | sed 's/\//-/g')
MODEL_ALIAS="${MODEL_ALIAS:-$DEFAULT_MODEL_ALIAS}"
# Get number of GPUs available
USE_JUDGE="${USE_JUDGE:-False}"
USE_API="${USE_API:-False}"
USE_PRIVILEGED="${USE_PRIVILEGED:-False}"
USE_REWARD_FN="${USE_REWARD_FN:-False}"
USE_JUDGE_ALT="${USE_JUDGE_ALT:-False}"
NUM_GPUS=$(nvidia-smi -L | wc -l)
USE_GCS="${USE_GCS:-False}"
N_ROLLOUTS="${N_ROLLOUTS:-8}"
N_COMPARES="${N_COMPARES:-7}"
RUN_NUMBER="${RUN_NUMBER:-0}"
FUNCTION_NAME="${FUNCTION_NAME:-compute_score}"
FUNCTION_PATH="${FUNCTION_PATH:-reward_fn}"
RUN_LABEL="${RUN_LABEL:-grpo}"
RUN_NAME=${RUN_LABEL}+${MODEL_ALIAS}+${TASK}+${LANGUAGE}+${RUN_NUMBER}
DEFAULT_FULL_DATA_PATH=${DATA_PATH}prep_traces/${TASK}+${LANGUAGE}/
FULL_DATA_PATH="${FULL_DATA_PATH:-$DEFAULT_FULL_DATA_PATH}"
FULL_SAVE_PATH=${SAVE_MODEL_PATH}${RUN_NAME}
DEBUG="${DEBUG:-False}"
JUDGE="${JUDGE:-azure/o4-mini}"
MAX_QUERY_LENGTH=4096
MAX_RESPONSE_LENGTH=2048
TRAIN_BS=32
LOGPROB_BS=32
PPO_BS=8

echo $RUN_NAME
echo $FUNCTION_NAME
python -m sp3f.main_grpo \
    +trainer.lang_code=${LANGUAGE} \
    +trainer.task=${TASK} \
    +trainer.use_gcs=${USE_GCS} \
    +trainer.gcs_project=${GCS_PROJECT} \
    +trainer.gcs_token=${GCS_TOKEN} \
    +trainer.gcs_path=${GCS_PATH}${RUN_NAME} \
    +trainer.use_judge=${USE_JUDGE} \
    +trainer.use_reward_fn=${USE_REWARD_FN} \
    +trainer.use_privileged=${USE_PRIVILEGED} \
    +trainer.judge_model=${JUDGE} \
    +trainer.use_judge_alt=${USE_JUDGE_ALT} \
    +trainer.debug=${DEBUG} \
    +trainer.use_api_judge=${USE_API} \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=${FULL_DATA_PATH}/train.parquet \
    data.val_files=${FULL_DATA_PATH}/test.parquet \
    data.prompt_key=input \
    data.train_batch_size=${TRAIN_BS} \
    data.max_prompt_length=${MAX_QUERY_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.shuffle=True \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    actor_rollout_ref.model.path=${MODEL_PATH}${MODEL} \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BS} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_BS} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-sum-norm" \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGPROB_BS} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${N_ROLLOUTS} \
    actor_rollout_ref.rollout.max_num_batched_tokens=20480 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    +actor_rollout_ref.rollout.compare=${N_COMPARES} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOGPROB_BS} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='lbr-lang_boot' \
    trainer.experiment_name=${RUN_NAME} \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.balance_batch=False \
    trainer.save_freq=250 \
    trainer.test_freq=500 \
    trainer.total_epochs=20 \
    trainer.total_training_steps=2005 \
    trainer.default_local_dir=${FULL_SAVE_PATH}/checkpoints/ \
    trainer.validation_data_dir=${FULL_SAVE_PATH}/evaluations/ \
    custom_reward_function.path=lang_boot/lang_boot/reward_functions/${FUNCTION_PATH}.py \
    custom_reward_function.name=${FUNCTION_NAME} \
    ${OTHER_ARGS}
