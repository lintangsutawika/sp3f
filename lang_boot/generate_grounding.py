import os
import random

import numpy as np
import pandas as pd
from datasets import load_dataset

def highest_loglikelihood(dataset):

    def get_most_likely_answer(example):
        example["query"] = example["step"][0]["full_input"][-1]["content"]
        example["input"] = example["answer"][example["logprob"].index(max(example["logprob"]))]
        example["accuracy"] = example["accuracy"][example["logprob"].index(max(example["logprob"]))]
        example["output"] = example["ground_truth"]
        return example

    unused_columns = ['sample_id', 'total_step', 'task_step', 'step', 'current_loop', 'logprob', 'answer']

    dataset = dataset.map(get_most_likely_answer, remove_columns=unused_columns)
    return dataset

df_en = highest_loglikelihood(load_dataset("json", data_files="data/deepscaler_en_generated.jsonl", split="train")).to_pandas()
df_id_generated = load_dataset("json", data_files="data/deepscaler_id_generated.jsonl", split="train").to_pandas()
df_id_translated = load_dataset("json", data_files="data/deepscaler_id_translated.jsonl", split="train").to_pandas()


data_df = pd.DataFrame()
# query = df_id_generated.iloc[0]['step'][0]['full_input'][-1]['content']
en_idx = df_en[df_en['accuracy'] == 1]["idx"].tolist()

use_df_id_generated = df_id_generated[df_id_generated['idx'].isin(en_idx)]
use_df_id_generated["query"] = use_df_id_generated.apply(lambda x: x['step'][0]['full_input'][-1]['content'], axis=1)
use_df_id_translated = df_id_translated[df_id_translated['idx'].isin(en_idx)]

def get_answer(x, right=True):
    accuracy = x['accuracy']
    answer = x['answer']
    answer_list = [a for i, a in enumerate(answer) if accuracy[i] == right]
    if len(answer_list) == 0:
        return None
    return random.choice(answer_list)

def get_translation(x):
    accuracy = x['accuracy']
    answer = x['answer']
    lang = x['lang']
    score = [x*y for x, y in zip(accuracy, lang)]
    return answer[score.index(max(score))]

data_path = "lang_boot/tasks/task_evaluation/judge_grounding/"

# En traces, X generated traces Right and X translated Traces
data_df['original'] = df_en[df_en['idx'].isin(en_idx)]['input']
# data_df['query'] = use_df_id_generated['query'].values
data_df['query'] = df_en[df_en['idx'].isin(en_idx)]['query']
data_df['yes'] = use_df_id_translated.apply(lambda x: get_translation(x), axis=1)
data_df['no'] = use_df_id_generated.apply(lambda x: get_answer(x, right=True), axis=1)
data_df.to_json(os.path.join(data_path, "g_right_v_t.jsonl"), orient="records", lines=True)

# En traces, X generated traces Wrong and X translated Traces
data_df['original'] = df_en[df_en['idx'].isin(en_idx)]['input']
# data_df['query'] = use_df_id_generated['query'].values
data_df['query'] = df_en[df_en['idx'].isin(en_idx)]['query']
data_df['yes'] = use_df_id_translated.apply(lambda x: get_translation(x), axis=1)
data_df['no'] = use_df_id_generated.apply(lambda x: get_answer(x, right=False), axis=1)
data_df.to_json(os.path.join(data_path, "g_wrong_v_t.jsonl"), orient="records", lines=True)

# En traces, X generated traces Right and X generated Traces Wrong
data_df['original'] = df_en[df_en['idx'].isin(en_idx)]['input']
# data_df['query'] = use_df_id_generated['query'].values
data_df['query'] = df_en[df_en['idx'].isin(en_idx)]['query']
data_df['yes'] = use_df_id_generated.apply(lambda x: get_answer(x, right=True), axis=1)
data_df['no'] = use_df_id_generated.apply(lambda x: get_answer(x, right=False), axis=1)
data_df.to_json(os.path.join(data_path, "g_right_v_g_wrong.jsonl"), orient="records", lines=True)

# En traces, X generated traces En and X generated Traces x Right
data_df['original'] = df_en[df_en['idx'].isin(en_idx)]['input']
# data_df['query'] = use_df_id_generated['query'].values
data_df['query'] = df_en[df_en['idx'].isin(en_idx)]['query']
data_df['yes'] = use_df_id_generated.apply(lambda x: get_answer(x, right=True), axis=1)
data_df['no'] = df_en[df_en['idx'].isin(en_idx)]['input']
data_df.to_json(os.path.join(data_path, "g_right_v_en_right.jsonl"), orient="records", lines=True)

# En traces, X generated traces En and X translated Traces
data_df['original'] = df_en[df_en['idx'].isin(en_idx)]['input']
# data_df['query'] = use_df_id_generated['query'].values
data_df['query'] = df_en[df_en['idx'].isin(en_idx)]['query']
data_df['yes'] = use_df_id_translated.apply(lambda x: get_translation(x), axis=1)
data_df['no'] = df_en[df_en['idx'].isin(en_idx)]['input']
data_df.to_json(os.path.join(data_path, "t_v_en_right.jsonl"), orient="records", lines=True)
