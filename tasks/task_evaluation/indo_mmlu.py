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
    # return f"{x['question']}\nA){x['option_a']}\nB){x['option_b']}\nC){x['option_c']}\nD){x['option_d']}\n"
    options = '\n'.join(eval(x['options']))
    return f"{x['question']}\n{options}\n\n"

def output_text(x):
    options = eval(x['options'])
    try:
        _, *answer = options[["A", "B", "C", "D", "E"].index(x["answer"].strip())].split(".")
        answer = ".".join(answer)
        answer = answer.strip()
    except:
        answer = "NONE"

    label = x["answer"].lower()
    return f"{label}:::{answer}"

def eval_with_postprocessing(x, y):
    label, answer = y.split(":::")

    ans = get_boxed_answer(x)
    ans = extract_text_content(ans).lower()
    if ans == label:
        return 1
    elif ans == answer.lower():
        return 1
    elif ans[:1] == label.lower():
        return 1
    return 0

@register_task("indo_mmlu_id")
class IndoMMLUTask(YevalTask):
    data_path="indolem/IndoMMLU"
    input_text=input_text
    output_text=output_text
    test_split="test"
    evaluation={"accuracy": eval_with_postprocessing}
    sample_agg_fn={"accuracy": lambda x: x}
    logging=log_logprob
