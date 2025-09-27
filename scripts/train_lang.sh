while getopts ":m:l:t:s:v:g:" opt; do
  case ${opt} in
    m ) MODEL=$OPTARG;;
    l ) LANGUAGE=$OPTARG;;
    t ) TASK=$OPTARG;;
    s ) SAVE_MODEL_PATH=$OPTARG;;
    v ) RUN_NUMBER=$OPTARG;;
    g ) USE_GCS=$OPTARG;;
  esac
done

# SAVE_MODEL_PATH=/data/user_data/lsutawik/lbr-language_bootstrap_reasoning/
# SAVE_MODEL_PATH=/scratch/lsutawik/
# SAVE_MODEL_PATH=/dev/shm/lsutawik/

MODEL=Qwen/Qwen3-4B
MODEL=Qwen/Qwen2.5-7B
TASK=deepscaler_train
SAVE_MODEL_PATH=/datadrive/lsutawik/lbr/models/
USE_GCS=False
LANGUAGE=id
RUN_NUMBER=0

sbatch lang_boot/scripts/train_grpo.sh \
  -r r_acc-penalize_lang -v ${RUN_NUMBER} \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -z /datadrive/lsutawik/lbr/data/deepscaler-${LANGUAGE}-q2.5-7b/ -s ${SAVE_MODEL_PATH} \
  -u reward_fn_q3 -f compute_score_reward_acc_penalize_lang

bash lang_boot/scripts/train_grpo.sh \
  -r r_acc-r_penalize_en -v ${RUN_NUMBER} \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -z /datadrive/lsutawik/lbr/data/deepscaler-${LANGUAGE}-q2.5-7b/ -s ${SAVE_MODEL_PATH} \
  -f compute_score_reward_acc_add_penalize_en

bash lang_boot/scripts/train_grpo.sh \
  -r r_acc-a-r_lang_fn -v ${RUN_NUMBER} \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -z /datadrive/lsutawik/lbr/data/deepscaler-${LANGUAGE}-q2.5-7b/ -s ${SAVE_MODEL_PATH} \
  -f compute_score_reward_acc_add_lang_fn


MODEL=Qwen/Qwen3-0.6B
MODEL=Qwen/Qwen3-4B
MODEL=Qwen/Qwen2.5-7B
TASK=deepscaler_train
SAVE_MODEL_PATH=/data/user_data/lsutawik/lbr-language_bootstrap_reasoning/
USE_GCS=False
LANGUAGE=id
RUN_NUMBER=0

sbatch lang_boot/scripts/train_grpo.sh \
  -r r_privileged-r_acc-r_penalize_en -v ${RUN_NUMBER} \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -z /data/user_data/lsutawik/lbr-language_bootstrap_reasoning/data/deepscaler-${LANGUAGE}-q2.5-7b/ -s ${SAVE_MODEL_PATH} -g $USE_GCS \
  -f compute_score_reward_acc_add_penalize_en -j True -p True -w True -b True

sbatch lang_boot/scripts/train_grpo.sh \
  -r r_privileged_API_GPT_5-r_acc-r_penalize_en -v ${RUN_NUMBER} \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -z /data/user_data/lsutawik/lbr-language_bootstrap_reasoning/data/deepscaler-${LANGUAGE}-q2.5-7b/ -s ${SAVE_MODEL_PATH} \
  -f compute_score_reward_acc_add_penalize_en \
  -j True -p True -w True -b True -i True -k azure/gpt-5

sbatch lang_boot/scripts/train_grpo.sh \
  -r r_acc-penalize_lang -v ${RUN_NUMBER} \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -z /data/user_data/lsutawik/lbr-language_bootstrap_reasoning/data/deepscaler-${LANGUAGE}-q2.5-7b/ -s ${SAVE_MODEL_PATH} \
  -u reward_fn_q3 -f compute_score_reward_acc_penalize_lang

sbatch lang_boot/scripts/train_grpo.sh \
  -r r_acc -v ${RUN_NUMBER} \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -z /data/user_data/lsutawik/lbr-language_bootstrap_reasoning/data/deepscaler-${LANGUAGE}-q2.5-7b/ -s ${SAVE_MODEL_PATH} \
  -u reward_fn_q3 -f compute_score_reward_acc


MODEL=Qwen/Qwen2.5-7B
TASK=deepscaler_train
SAVE_MODEL_PATH=/data/user_data/lsutawik/lbr-language_bootstrap_reasoning/
USE_GCS=False
LANGUAGE=id
RUN_NUMBER=0


MODEL=/datadrive/lsutawik/lbr/models/sft-os+Qwen-Qwen2.5-7B+deepscaler_train+en+0/huggingface
TASK=deepscaler_train
USE_GCS=False
LANGUAGE=id
RUN_NUMBER=0

bash lang_boot/scripts/train_grpo.sh \
  -r r_acc-r_penalize_en -v ${RUN_NUMBER} \
  -a sft-Qwen2.5-7B-deepscaler_train \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -z /datadrive/lsutawik/lbr/data/deepscaler-${LANGUAGE}-q2.5-7b/ -s ${SAVE_MODEL_PATH} \
  -f compute_score_reward_acc_add_penalize_en

# SAVE_MODEL_PATH=/scratch/lsutawik/
# USE_GCS=True

# r_rand
sbatch lang_boot/scripts/train_grpo.sh \
  -r r_rand -v ${RUN_NUMBER} \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -d data/ -s ${SAVE_MODEL_PATH} -g $USE_GCS \
  -f compute_score_reward_rand

# r_acc
sbatch lang_boot/scripts/train_grpo.sh \
  -r r_acc -v ${RUN_NUMBER} \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -d data/Qwen-Qwen2.5-7B/ -s ${SAVE_MODEL_PATH} -g $USE_GCS \
  -f compute_score_reward_acc


sbatch lang_boot/scripts/train_grpo.sh \
  -r r_acc+r_sim -v ${RUN_NUMBER} \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -d data/Qwen-Qwen2.5-7B/ -s ${SAVE_MODEL_PATH} -g $USE_GCS \
  -f compute_score_reward_acc -o "+use_similarity=True"

# r_acc-a-r_lang_fn
sbatch lang_boot/scripts/train_grpo.sh \
  -r r_acc-a-r_lang_fn -v ${RUN_NUMBER} \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -d data/Qwen-Qwen2.5-7B/ -s ${SAVE_MODEL_PATH} -g $USE_GCS \
  -f compute_score_reward_acc_add_lang_fn

# r_acc-a-r_lang_fn-w-rmse
sbatch lang_boot/scripts/train_grpo.sh \
  -r r_acc-a-r_lang_fn-w-rmse -v ${RUN_NUMBER} \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -d /data/user_data/lsutawik/lbr-language_bootstrap_reasoning/data/Qwen-Qwen2.5-7B/ -s ${SAVE_MODEL_PATH} -g $USE_GCS \
  -f compute_score_reward_acc_add_lang_fn_with_rmse \
  -o "actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra']"


MODEL=Qwen/Qwen2.5-7B
TASK=deepscaler_train
SAVE_MODEL_PATH=/data/user_data/lsutawik/lbr-language_bootstrap_reasoning/
LANGUAGE=id
RUN_NUMBER=0

# bash lang_boot/scripts/train_grpo.sh \
sbatch lang_boot/scripts/train_grpo.sh \
  -r r_privileged-r_penalize_en-API -v ${RUN_NUMBER} \
  -a test-model \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -z /data/user_data/lsutawik/lbr-language_bootstrap_reasoning/data/deepscaler-${LANGUAGE}-q2.5-7b/ -s ${SAVE_MODEL_PATH} \
  -f compute_score_reward_acc_add_penalize_en -j True -p True -i True -w True


MODEL=Qwen/Qwen2.5-7B

MODEL=/datadrive/lsutawik/lbr/models/sft-os+Qwen-Qwen2.5-7B+deepscaler_train+en+0/huggingface

MODEL=/data/user_data/lsutawik/lbr-language_bootstrap_reasoning/sft-os+Qwen-Qwen2.5-7B+deepscaler_train+en+0/checkpoints/global_step_500/huggingface
TASK=deepscaler_train
SAVE_MODEL_PATH=/data/user_data/lsutawik/lbr-language_bootstrap_reasoning/
# SAVE_MODEL_PATH=/datadrive/lsutawik/lbr/models/
USE_GCS=False
LANGUAGE=id
RUN_NUMBER=0

sbatch lang_boot/scripts/train_grpo.sh \
  -r r_privileged_API-r_acc-r_penalize_en -v ${RUN_NUMBER} \
  -a sft-Qwen2.5-7B-deepscaler_train \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -z /data/user_data/lsutawik/lbr-language_bootstrap_reasoning/data/deepscaler-${LANGUAGE}-q2.5-7b/ -s ${SAVE_MODEL_PATH} \
  -f compute_score_reward_acc_add_penalize_en \
  -j True -p True -w True -b True -i True


bash lang_boot/scripts/train_grpo.sh \
  -r r_privileged_API-r_acc-r_penalize_en -v ${RUN_NUMBER} \
  -a sft-Qwen2.5-7B-deepscaler_train \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -z /datadrive/lsutawik/lbr/data/deepscaler-${LANGUAGE}-q2.5-7b/ -s ${SAVE_MODEL_PATH} \
  -f compute_score_reward_acc_add_penalize_en -j True -p True -w True -i True -b True

sbatch lang_boot/scripts/train_grpo.sh \
  -r r_acc-r_penalize_en -v ${RUN_NUMBER} \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -z /data/user_data/lsutawik/lbr-language_bootstrap_reasoning/data/deepscaler-${LANGUAGE}-q2.5-7b/ -s ${SAVE_MODEL_PATH} -g $USE_GCS \
  -f compute_score_reward_acc_add_penalize_en \
  -o "actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra']"

  -a sft-Qwen2.5-7B-deepscaler_train \


sbatch lang_boot/scripts/train_grpo.sh \
  -r r_privileged-r_acc-r_penalize_en -v ${RUN_NUMBER} \
  -a sft-Qwen2.5-7B-deepscaler_train \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -z /data/user_data/lsutawik/lbr-language_bootstrap_reasoning/data/deepscaler-${LANGUAGE}-q2.5-7b/ -s ${SAVE_MODEL_PATH} -g $USE_GCS \
  -f compute_score_reward_acc_add_penalize_en -j True -p True -w True -b True \
  -o "actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model','optimizer','extra']"

# Run next
sbatch lang_boot/scripts/train_grpo.sh \
  -r r_privileged-r_acc-r_penalize_en-os -v ${RUN_NUMBER} \
  -a sft-Qwen2.5-7B-deepscaler_train \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -z /data/user_data/lsutawik/lbr-language_bootstrap_reasoning/data/deepscaler_os-${LANGUAGE}-q2.5-7b/ -s ${SAVE_MODEL_PATH} -g $USE_GCS \
  -f compute_score_reward_acc_add_penalize_en -p True -j True -w True -b True -i True 

# r_privileged
sbatch lang_boot/scripts/train_grpo.sh \
  -r r_privileged -v ${RUN_NUMBER} \
  -a sft-Qwen2.5-7B-deepscaler_train \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -d /data/user_data/lsutawik/lbr-language_bootstrap_reasoning/data/Qwen-Qwen2.5-7B/ -s ${SAVE_MODEL_PATH} -g $USE_GCS \
  -f compute_score_reward_acc -p True -j True -b True

sbatch lang_boot/scripts/train_grpo.sh \
  -r r_privileged-r_parsable -v ${RUN_NUMBER} \
  -a sft-Qwen2.5-7B-deepscaler_train \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -d /data/user_data/lsutawik/lbr-language_bootstrap_reasoning/data/Qwen-Qwen2.5-7B/ -s ${SAVE_MODEL_PATH} -g $USE_GCS \
  -f compute_score_reward_parseable -p True -j True -w True


# r_acc-m-r_lang_fn
sbatch lang_boot/scripts/train_grpo.sh \
  -r r_acc-m-r_lang_fn -v ${RUN_NUMBER} \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -d data/ -s ${SAVE_MODEL_PATH} -g $USE_GCS \
  -f compute_score_reward_acc_mult_lang_fn -e translated

# r_privileged
sbatch lang_boot/scripts/train_grpo.sh \
  -r r_privileged -v ${RUN_NUMBER} \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -d /data/user_data/lsutawik/lbr-language_bootstrap_reasoning/data/Qwen-Qwen2.5-7B/ -s ${SAVE_MODEL_PATH} -g $USE_GCS \
  -f compute_score_reward_acc -p True -j True -b True
  #  -e translated


# r_privileged-r_parsable
sbatch lang_boot/scripts/train_grpo.sh \
  -r r_privileged-r_parsable -v ${RUN_NUMBER} \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -d /data/user_data/lsutawik/lbr-language_bootstrap_reasoning/data/Qwen-Qwen2.5-7B/ -s ${SAVE_MODEL_PATH} -g $USE_GCS \
  -f compute_score_reward_parseable -p True -j True -w True

# r_acc-r_privileged
sbatch lang_boot/scripts/train_grpo.sh \
  -r r_acc-r_privileged -v ${RUN_NUMBER} \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -d data/Qwen-Qwen2.5-7B/ -s ${SAVE_MODEL_PATH} -g $USE_GCS \
  -f compute_score_reward_acc -p True -j True -w True

# r_acc-r_lang_lm
sbatch lang_boot/scripts/train_grpo.sh \
  -r r_acc-r_lang_lm -v ${RUN_NUMBER} \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -d data/Qwen-Qwen2.5-7B/ -s ${SAVE_MODEL_PATH} -g $USE_GCS \
  -f compute_score_reward_acc -p False -j True -w True

# for STEP in 50 100 150 200 250 300 350 400 450
# do
#     find ./global_step_${STEP}/ -name "*.pt" -delete
# done




# RUN_NUMBER=0
# MODEL=Qwen/Qwen2.5-7B-Instruct
# LANGUAGE=ja
# TASK=gsm8k_train
# SAVE_MODEL_PATH=/data/user_data/lsutawik/lbr-language_bootstrap_reasoning/
# STEP=500

# bash lang_boot/scripts/eval_model.sh \
#   -r r_acc-m-r_lang_fn -v ${RUN_NUMBER} \
#   -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
#   -d data/ -s ${SAVE_MODEL_PATH} -g $USE_GCS \
#   -p ${STEP} \
#   -f compute_score_reward_acc_mult_lang_fn




# import re

# def extract_text_content(text):
#     match = re.search(r'\\text\{([^}]+)\}', text)
#     return match.group(1) if match else text


# def rescore(row):
#     task_eval = row["data_source"]
#     if task_eval == "train_dataset":
#         return row["task_score"]
#     ans = extract_text_content(row["ans"])
#     ans = "\\boxed{"+ans+"}" 
#     ground_truth = row["gold"]
#     score = eval_fn[task_eval](ans, ground_truth)["accuracy"]
#     return score

# df['task_score'] = df.apply(rescore, axis=1)

