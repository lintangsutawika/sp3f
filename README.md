# 🏖️ SP3F: Self-Play with Privileged Pairwise Feedback

Official implementation of SP3F, method from "Gained In Translation: Privileged Pairwise Judges Enhance Multilingual Reasoning".

by [Lintang Sutawika](https://lintang.sutawika.com/), [Gokul Swamy](https://gokul.dev/), [Zhiwei Steven Wu](https://zstevenwu.com/), and [Graham Neubig](https://www.phontron.com/)

<p align="center">
  <img width="800" src="assets/sp3f_ffig.png">
</p>

We introduce Self-Play with Privileged Pairwise Feedback (SP3F), a two-stage framework for
enhancing multilingual reasoning without any data in the target language(s). First, we supervise fine-tune (SFT) on translated versions of English question-answer pairs to raise base model correctness. Second, we perform RL with feedback from a pairwise judge in a self-play fashion. Our key insight is that we can use English reference responses during both SFT and RL by framing both learning problems in terms of translation. We use reference responses as data for translation during SFT and as privileged information for the pairwise judge during downstream RL. 

<p align="center">
  <img width="400" src="assets/sp3f_pareto.png">
</p>

We apply SP3F on data from 18 languages and find that RLMs trained via SP3F outperform fully post-trained models such as Qwen2.5-7B-Instruct on both in-domain math and out-of-domain non-math tasks in a target language while using 1/8th as much training data. We also find particularly large improvements on lower-resourced languages and see better generalization to unseen languages. Our experiments show that privileged information is particularly helpful in improving detection of correct reasoning chains, even if the final answer is incorrect.


[![arXiv](https://img.shields.io/badge/arXiv-2601.18722-df2a2a.svg?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2601.18722)
[![HF Collection](https://img.shields.io/badge/%F0%9F%A4%97-Datasets-yellow?style=for-the-badge)](https://huggingface.co/collections/neulab/sp3f)
[![License](https://img.shields.io/github/license/TRI-ML/prismatic-vlms?style=for-the-badge)](LICENSE)
[![Summary](https://img.shields.io/badge/ -Summary-1DA1F2?logo=x&logoColor=white&labelColor=gray&style=for-the-badge)]()

## ⚙️ Setup

We recommend using GPUs with at least 48GB of memory. Our experiments were run on 8xL40s for a single training experiment. 

To get started, 
```
git clone https://github.com/lintangsutawika/SP3F.git
cd SP3F/
```

And setup an `.env` to hold API keys for `LLM_API_URL` and `LLM_API_KEY` to configure the llm-judge used to train SP3F.
```
export LLM_API_URL=...
export LLM_API_KEY=...
```

Our scripts uses [uv](https://docs.astral.sh/uv/) to handle dependancies. All dependancies are defined in `pyproject.toml`

## 🏋🏽 Training SP3F

You can download the data we used from [neulab/SP3F-Training-Data](https://huggingface.co/datasets/neulab/SP3F-Training-Data).
```
git clone https://huggingface.co/datasets/neulab/SP3F-Training-Data
```

SP3F consists of 2 stages, an initial SFT stage and a GRPO stage with privileged information.

### 1️⃣ SFT

To start, we'll finetune a base model using [scripts/train_sft.sh](scripts/train_sft.sh). This is to align the model with a starting distribution that mimics intended behaviours such as proper format and target language to some degree.

```
MODEL=Qwen/Qwen2.5-7B
LANGUAGE=all
DATA_PATH=/absolute/path/to/SP3F-Training-Data/
SAVE_PATH=...
sbatch scripts/train_sft.sh \
    -m ${MODEL} \
    -l ${LANGUAGE} \
    -f ${DATA_PATH} \
    -s ${SAVE_PATH}
```

### 2️⃣ GRPO with Privileged Information

Next, we train our SFT model with RL. It is key to use a capable LLM-as-a-Judge. Our experiments use GPT-4o-Mini but other LLMs may work as well. Specifically, we use [scripts/train_grpo.sh](scripts/train_grpo.sh)

```
TASK=deepscaler_train
LANGUAGE=all
MODEL=...
RUN_NAME=r_privileged_API_GPT_4o_Mini-r_acc-r_threshold-r_parsable
DATA_PATH=/absolute/path/to/SP3F-Training-Data/
SAVE_PATH=...
USE_JUDGE=True
USE_PRIVILEGED_INFO=True
USE_RLVR=True
USE_API_JUDGE=True
API_MODEL=... (OpenAI API-compatible model)
export LLM_API_URL=...
export LLM_API_KEY=...
sbatch scripts/train_grpo.sh \
  -r ${RUN_NAME} \
  -m ${MODEL} -l ${LANGUAGE} -t ${TASK} \
  -z ${DATA_PATH} -s ${SAVE_PATH} \
  -f compute_score_reward_acc_add_parseable_add_threshold \
  -j ${USE_JUDGE} -p ${USE_PRIVILEGED_INFO} -w ${USE_RLVR} -i ${USE_API_JUDGE} -k ${API_MODEL}
```

### 💬 Adding new languages

To add more languages, you can use the our scripts to translate English data to your target language.
#### Register a translation prompt. 

Add a new class in [tasks/translate_prompt.py](tasks/translate_prompt.py) this will look something like this. Register it using `<Language ID>_translate` format. We will use the language id to identify which prompt to use during translation.

```
@register_task("id_translate")
class IDTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Indonesian. \
Respond directly after \"Indonesian Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"{x}"+"Indonesian Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="id")}
```

#### Translate Queries and Solutions

Setup your LLM which could be a locally hosted one or an API model and set their base_url and key in LLM_URL and LLM_KEY in `.env` the translation scripts will look for this file.

To translate queries, we use [scripts/reasoning_translate_queries.sh](scripts/reasoning_translate_queries.sh) with the following parameters:
```
MODEL=azure/gpt-5-nano # We use GPT-5-Nano for our translation
LANGUAGE=id # Language code 
SAVE_PATH=...
MAX_SAMPLES=5000 # Number of samples to be translated
bash scripts/reasoning_translate_queries.sh \
            -t deepscaler_train \
            -m ${MODEL} \
            -l ${LANGUAGE} \
            -y ${SAVE_PATH} \
            -o "--n_samples ${MAX_SAMPLES}"
```

And to translate solutions, we use [scripts/reasoning_translate_solutions.sh](scripts/reasoning_translate_solutions.sh).
```
MODEL=azure/gpt-5-nano # We use GPT-5-Nano for our translation
LANGUAGE=id # Language code 
SAVE_PATH=...
MAX_SAMPLES=5000 # Number of samples to be translated
bash scripts/reasoning_translate_solutions.sh \
            -t deepscaler_train \
            -m ${MODEL} \
            -l ${LANGUAGE} \
            -y ${SAVE_PATH} \
            -o "--n_samples ${MAX_SAMPLES}"
```

#### Construct Training Dataset

With the translated queries and solutions we can [construct](src/construct_dataset.py) the training data.
```
uv run --isolated src/construct_dataset.py \
        --data_path ${SAVE_PATH} \
        --output_path ${OUTPUT_PATH} \
        --lang ${LANGUAGE} \
        --use_en_solution \
        --use_translated_solution
```

### 📝 Training on custom tasks

To use your own data.

1. `solution`: English solution
2. `translated_solution`: Equivalent to the English solution translated to a target language.
3. `reward_model`: For the reward model to use. Consist of dict  with field `ground_truth`, example: `{"ground_truth": "2\\sqrt{3} - 1"}`
4. `input`: dict that contains the system and user prompt. Example: `[{"role": "system", "content": ...}, {"role": "user", "content": ...}]`
5. `extra_info`: Auxilary information used for the RLVR, `{"ground_truth": "2\\sqrt{3} - 1", "lang": <Language Code>}`


### 🔍 Evaluating models

To evaluate trained or baseline models, we use the [scripts/eval_task.sh](scripts/eval_task.sh). Each run processes a single language for all 4 tasks, so we will need to iterate over all of the languages. Tip: We use SLURM which allows us to parallelize the evals for each language.

```
# Path to a local model HF checkpoint. HF model names such as Qwen/Qwen2.5-7B also work.
MODEL_PATH=...
# Simplified or specific name or your model. Will default to MODEL_PATH if not set.
MODEL_ALIAS=...
SAVE_PATH=...
for LANGUAGE in ar bn de en es fr hi id it ja ko pt ru sw te th yo zh
do
    bash scripts/eval_task.sh \
        -m ${MODEL_PATH} \
        -a ${MODEL_ALIAS} \
        -s ${SAVE_PATH} \
        -l ${LANGUAGE}
done
```

Specifically for `MT MATH 100` in Chinese (zh). The dataset features distinct Mainland Chinese (cn) and Taiwanese Chinese (tw). We report the averaged score over these runs in our paper. To evaluate them, use [scripts/eval_task_math100_zh.sh](scripts/eval_task_math100_zh.sh)
```
# Path to a local model HF checkpoint. HF model names such as Qwen/Qwen2.5-7B also work.
MODEL_PATH=...
# Simplified or specific name or your model. Will default to MODEL_PATH if not set.
MODEL_ALIAS=...
SAVE_PATH=...
bash scripts/eval_task_math100_zh.sh \
    -m ${MODEL_PATH} \
    -a ${MODEL_ALIAS} \
    -s ${SAVE_PATH} \
    -l ${LANGUAGE}
```

The output eval will be an `output.jsonl` with the following fields:
1. `lang`: Percentage of the response is in the target language (We then convert this to language fidelity of 0/1 if this percentage is equal or greater than a threshold.
2. `accuracy`: Whether the final answer is correct according to the ground truth.
3. `ground_truth`: The correct answer from the dataset.
4. `answer`: Model response.

## 📚 Citation

If you found this work useful, please cite us by using the following bibtex
```
@misc{sutawika2026gainedtranslationprivilegedpairwise,
      title={Gained in Translation: Privileged Pairwise Judges Enhance Multilingual Reasoning}, 
      author={Lintang Sutawika and Gokul Swamy and Zhiwei Steven Wu and Graham Neubig},
      year={2026},
      eprint={2601.18722},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.18722}, 
}
```

## 🎉 Acknowlegedments

Our codebase utilizes [Verl](https://github.com/volcengine/verl).
