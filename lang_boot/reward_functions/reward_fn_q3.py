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

def parse_response(response):
    response_format = r"<think>(.*?)</think>\s*(.*)"
    try:
        format_matched = re.findall(response_format, response, re.DOTALL)
    except Exception as e:
        return "", response

    if format_matched:
        return format_matched[0]
    else:
        return "", ""

def compute_score(data_source, solution_str, ground_truth, extra_info=None,
                  use_penalty=False,
                  use_random=False,
    ):

    thinking_str, answer_str = parse_response(solution_str)

    task_eval = extra_info["task"].split("/")[0]
    if task_eval == "train_dataset":
        gold = ground_truth
        ans = get_boxed_answer(answer_str)
        if ans == "None":
            ans_score = 0
        else:
            ans_score = math_eval(ans, gold)
    else:
        ans = get_boxed_answer(answer_str)
        ans_score = eval_fn[task_eval](answer_str, ground_truth)["accuracy"]

    lang = extra_info["lang"]
    _, answer_lang_score, answer_en_score = get_lang_score(answer_str, lang=lang, check_en=True)
    _, thinking_lang_score, thinking_en_score = get_lang_score(thinking_str, lang=lang, check_en=True)

    if use_random:
        reward = random.random() >= 0.5
    else:
        reward = ans_score
        if use_penalty:
            reward -= answer_en_score
            reward -= thinking_lang_score

    return {
        "data_source": extra_info["task"],
        "score": reward,
        "task_score": ans_score,
        "answer_lang_score": answer_lang_score,
        "answer_en_score": answer_en_score,
        "thinking_lang_score": thinking_lang_score,
        "thinking_en_score": thinking_en_score,
        "use_penalty": use_penalty,
        "use_random": use_random,
        "gold": ground_truth,
        "ans": ans,
        "parsable": 1 if str(ans) != None else 0,
    }

def compute_score_reward_rand(data_source, solution_str, ground_truth, extra_info):
    return compute_score(
        data_source, solution_str, ground_truth, extra_info,
        use_random=True,
    )

def compute_score_reward_acc(data_source, solution_str, ground_truth, extra_info):
    return compute_score(
        data_source, solution_str, ground_truth, extra_info,
        use_penalty=False,
    )

def compute_score_reward_acc_penalize_lang(data_source, solution_str, ground_truth, extra_info):
    return compute_score(
        data_source, solution_str, ground_truth, extra_info,
        use_penalty=True,
    )
