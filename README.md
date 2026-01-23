# 🏖️ Gained In Translation: Privileged Pairwise Judges Enhance Multilingual Reasoning

Official implementation of SP3F
by [Lintang Sutawika](https://lintang.sutawika.com/), [Gokul Swamy](https://gokul.dev/), [Zhiwei Steven Wu](https://zstevenwu.com/), and [Graham Neubig](https://www.phontron.com/)

<p align="center">
  <img width="1000" src="assets/sp3f_ffig.png">
</p>

We introduce Self-Play with Privileged Pairwise Feedback (SP3F), a two-stage framework for
enhancing multilingual reasoning without any data in the target language(s). First, we supervise fine-tune (SFT) on translated versions of English question-answer pairs to raise base model correctness. Second, we perform RL with feedback from a pairwise judge in a self-play fashion. Our key insight is that we can use English reference responses during both SFT and RL by framing both learning problems in terms of translation. We use reference responses as data for translation during SFT and as privileged information for the pairwise judge during downstream RL. 

<p align="center">
  <img width="500" src="assets/sp3f_pareto.png">
</p>

We apply SP3F on data from 18 languages and find that RLMs trained via SP3F outperform fully post-trained models such as Qwen2.5-7B-Instruct on both in-domain math and out-of-domain non-math tasks in a target language while using 1/8th as much training data. We also find particularly large improvements on lower-resourced languages and see better generalization to unseen languages. Our experiments show that privileged information is particularly helpful in improving detection of correct reasoning chains, even if the final answer is incorrect.


[![arXiv](https://img.shields.io/badge/arXiv-2506.05294-df2a2a.svg?style=for-the-badge&logo=arxiv)]()
[![HF Collection](https://img.shields.io/badge/%F0%9F%A4%97-Datasets-yellow?style=for-the-badge)](https://huggingface.co/collections/neulab/sp3f)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)
[![Summary](https://img.shields.io/badge/ -Summary-1DA1F2?logo=x&logoColor=white&labelColor=gray&style=for-the-badge)]()

## ⚙️ Setup

We recommend using GPUs with at least 48GB of memory. Our experiments were run on 8xL40s for a single training experiment. 

## 🏋🏽 Training SP3F

You can download the data we used from [neulab/SP3F-Training-Data](https://huggingface.co/datasets/neulab/SP3F-Training-Data).

SP3F consists of 2 stages, an initial SFT stage and a GRPO stage with privileged information.

### 1st Stage: SFT

To start, we finetune a base model.

```
MODEL=Qwen/Qwen2.5-7B
LANGUAGE=all
DATA_PATH=...
SAVE_PATH=...
sbatch lang_boot/scripts/train_sft.sh \
    -m ${MODEL} \
    -l ${LANGUAGE} \
    -f ${DATA_PATH} \
    -s ${SAVE_PATH}
```

### 2nd Stage: GRPO with Privileged Information
To train a model with SP3F, we use the following command. It is key to use a capable LLM-as-a-Judge. Our experiments use GPT-4o-Mini but other LLMs may work as well.

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

### Training on new tasks

To use your own data.

1. `solution`: English solution
2. `translated_solution`: Equivalent to the English solution translated to a target language.
3. `reward_model`: For the reward model to use. Consist of dict  with field `ground_truth`, example: `{"ground_truth": "2\\sqrt{3} - 1"}`
4. `input`: dict that contains the system and user prompt. Example: `[{"role": "system", "content": ...}, {"role": "user", "content": ...}]`
5. `extra_info`: Auxilary information used for the RLVR, `{"ground_truth": "2\\sqrt{3} - 1", "lang": <Language Code>}`

## 📚 Citation

If you found this work useful, please cite us by using the following bibtex
```
```

## 🎉 Acknowlegedments

This codebase heavily uses [Verl](https://github.com/volcengine/verl).
