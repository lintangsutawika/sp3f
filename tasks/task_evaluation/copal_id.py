import os
import re
from functools import partial

from yeval.task import register_task, YevalTask

from yeval.log.usage import log_token_usage
from yeval.response import (
    match_routing,
    preprocess_routing,
    postprocess_routing
    )

from yeval.metrics import math_eval
from yeval.log.usage import log_logprob
from yeval.response.math_responses import get_boxed_answer

from lang_boot.utils import extract_text_content
# from lang_boot.utils import math_eval_with_postprocessing

path = os.path.dirname(__file__)

def input_text(x):
    premise = x['premise']
    options = "A. " + x['choice1'] + "\nB. " + x['choice2']
    if x['question'] == "cause":
        return f"{premise} karena?\n{options}"
    elif x['question'] == "effect":
        return f"Jika {premise}, maka?\n{options}"

def output_text(x):
    label = ["A", "B"][x["label"]]
    answer = x[f"choice{x['label']+1}"]
    return f"{label}:::{answer}"

def eval_with_postprocessing(x, y):
    label, answer = y.split(":::")

    ans = get_boxed_answer(x)
    ans = extract_text_content(ans).lower()
    print(ans, y)
    if ans == label.lower():
        return 1
    elif ans == answer.lower():
        return 1
    elif ans[:1] == label.lower():
        return 1
    return 0

class COPALIDTask(YevalTask):
    data_path="haryoaw/COPAL"
    input_text=input_text
    output_text=output_text
    evaluation={"accuracy": eval_with_postprocessing}
    sample_agg_fn={"accuracy": lambda x: x}
    logging=log_logprob

@register_task("copal_id_standard_id")
class COPALIDStandardTask(COPALIDTask):
    test_split="test"

@register_task("copal_id_colloquial_id")
class COPALIDColloquialTask(COPALIDTask):
    test_split="test_colloquial"
