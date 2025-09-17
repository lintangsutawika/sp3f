import os
import re
import random

import numpy as np
from sklearn.metrics import mean_squared_error

from yeval.metrics import math_eval
from yeval.response.math_responses import get_boxed_answer
from lang_boot.utils import get_lang_score

from lang_boot.reward_functions.repetition_penalty import repetition_penalty

from yeval.task import TASK_LIST
from yeval.utils import import_modules

path = os.path.dirname(__file__)
import_modules(os.path.join(path, "../../tasks/"))

# Make dict for all tasks,
eval_tasks = ["mgsm", "global_mmlu", "belebele", "mt_aime2024", "mt_math100"]

eval_fn = {
    "math500": TASK_LIST["math500"]().eval,
    **{task: TASK_LIST[f"{task}_en"]().eval for task in eval_tasks}
}

def compute_score(data_source, solution_str, ground_truth, extra_info=None, use_lang=False, use_penalty=False, use_random=False, use_multiply=False, use_lang_threshold=False, use_parsable=False, penalize_english=False):

    task_eval = extra_info["task"].split("/")[0]
    if task_eval == "train_dataset":
        gold = ground_truth
        ans = get_boxed_answer(solution_str)
        if ans == "None":
            ans_score = 0
        else:
            ans_score = math_eval(ans, gold)
    else:
        ans = get_boxed_answer(solution_str)
        ans_score = eval_fn[task_eval](solution_str, ground_truth)["accuracy"]

    if use_random:
        reward = random.random() >= 0.5
    else:
        reward = ans_score

    if use_parsable:
        reward += 1 if ans else 0

    lang = extra_info["lang"]
    _, lang_score, en_lang_score = get_lang_score(solution_str, lang=lang, check_en=True)
    lang_reward = 0.0
    if use_lang_threshold:
        lang_penalty += np.sqrt(
            mean_squared_error(
                [use_lang_threshold], [lang_score]
            )
        )

    if use_lang and (lang != "en"):
        if use_lang_threshold:
            lang_reward = -lang_penalty
        else:
            lang_reward = lang_score

    if penalize_english:
        lang_reward -= en_lang_score

    if use_multiply:
        reward *= lang_reward
    else:
        reward += lang_reward

    penalty = 0
    N_gram_sizes = [2, 3, 4, 5]
    for N in N_gram_sizes:
        penalty += repetition_penalty(solution_str, window_size=N)
    penalty /= len(N_gram_sizes)

    if use_penalty and (lang != "en"):
        reward -= penalty

    return {
        "data_source": extra_info["task"],
        "score": reward,
        "task_score": ans_score,
        "lang_score": lang_score,
        "lang_penalty": lang_penalty if use_lang_threshold else 0,
        "repetition_penalty": penalty,
        "use_lang": use_lang,
        "use_penalty": use_penalty,
        "use_random": use_random,
        "gold": ground_truth,
        "ans": ans,
        "parsable": 1 if ans else 0,
        "use_lang_threshold": use_lang_threshold,
        "penalize_english": penalize_english,
    }

def compute_score_reward_acc(data_source, solution_str, ground_truth, extra_info):
    return compute_score(
        data_source, solution_str, ground_truth, extra_info,
        use_lang=False,
        use_penalty=True,
    )

def compute_score_reward_acc_add_lang_fn(data_source, solution_str, ground_truth, extra_info):
    return compute_score(
        data_source, solution_str, ground_truth, extra_info,
        use_lang=True,
        use_penalty=True,
    )

def compute_score_reward_acc_mult_lang_fn(data_source, solution_str, ground_truth, extra_info):
    return compute_score(
        data_source, solution_str, ground_truth, extra_info,
        use_lang=True,
        use_penalty=True,
        use_multiply=True,
    )

def compute_score_reward_rand(data_source, solution_str, ground_truth, extra_info):
    return compute_score(
        data_source, solution_str, ground_truth, extra_info,
        use_lang=False,
        use_penalty=False,
        use_random=True,
    )

def compute_score_reward_acc_add_lang_fn_with_rmse(data_source, solution_str, ground_truth, extra_info):
    return compute_score(
        data_source, solution_str, ground_truth, extra_info,
        use_lang=True,
        use_penalty=True,
        use_lang_threshold=0.65,
    )

def compute_score_reward_parseable(data_source, solution_str, ground_truth, extra_info):
    return compute_score(
        data_source, solution_str, ground_truth, extra_info,
        use_parsable=True,
    )

def compute_score_reward_parseable_add_penalize_en(data_source, solution_str, ground_truth, extra_info):
    return compute_score(
        data_source, solution_str, ground_truth, extra_info,
        use_parsable=True,
        penalize_english=True,
    )

def compute_score_reward_acc_add_penalize_en(data_source, solution_str, ground_truth, extra_info):
    return compute_score(
        data_source, solution_str, ground_truth, extra_info,
        penalize_english=True,
    )
