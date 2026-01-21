# Gained In Translation: Privileged Pairwise Judges Enhance Multilingual Reasoning

Official implementation of SP3F

<p align="center">
  <img width="1000" src="assets/sp3f_ffig.png">
</p>

by [Lintang Sutawika](https://lintang.sutawika.com/), [Gokul Swamy](https://gokul.dev/), [Steven Wu](https://zstevenwu.com/), and [Graham Neubig](https://www.phontron.com/)

[![arXiv](https://img.shields.io/badge/arXiv-2506.05294-df2a2a.svg?style=for-the-badge&logo=arxiv)]()
[![HF Collection](https://img.shields.io/badge/%F0%9F%A4%97-Datasets-yellow?style=for-the-badge)](https://huggingface.co/collections/neulab/sp3f)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)
[![Summary](https://img.shields.io/badge/ -Summary-1DA1F2?logo=x&logoColor=white&labelColor=gray&style=for-the-badge)]()

## Setup

Coming Soon

## Training SP3F

Coming Soon

### Training on new tasks

```
WORK_PATH=/data/user_data/lsutawik/lbr-language_bootstrap_reasoning/
RUN_NUMBER=0
TASK=deepscaler_train
LANGUAGE=all
MODEL=...
RUN_NAME=r_privileged_API_GPT_4o_Mini-r_acc-r_threshold-r_parsable
DATA_PATH=...
SAVE_PATH=...
USE_JUDGE=True
USE_PRIVILEGED_INFO=True
USE_RLVR=True
USE_API_JUDGE=True
API_MODEL=... (OpenAI API-compatible model)
export LLM_API_URL=...
export LLM_API_KEY=...
sbatch lang_boot/scripts/train_grpo.sh \
  -r ${RUN_NAME} \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -z ${DATA_PATH} -s ${SAVE_PATH} \
  -f compute_score_reward_acc_add_parseable_add_threshold \
  -j ${USE_JUDGE} -p ${USE_PRIVILEGED_INFO} -w {USE_RLVR} -i ${USE_API_JUDGE} -k ${API_MODEL}
```

## Citation

If you found this work useful, please cite us by using the following bibtex
```
```

## Acknowlegedments

This codebase heavily uses [Verl](https://github.com/volcengine/verl).
