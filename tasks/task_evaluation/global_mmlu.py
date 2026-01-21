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

# from lang_boot.utils import math_eval_with_postprocessing
from lang_boot.utils import (
    extract_text_content,
    highest_loglikelihood,
    highest_language_content,
    )

path = os.path.dirname(__file__)

def input_text(x):
    # return f"{x['question']}\nA){x['option_a']}\nB){x['option_b']}\nC){x['option_c']}\nD){x['option_d']}\n"
    return f"{x['question']}\n\nA) {x['option_a']}\nB) {x['option_b']}\nC) {x['option_c']}\nD) {x['option_d']}\n\n"

def output_text(x):
    label = x["answer"].lower()
    answer = x[f"option_{label}"]
    return f"{label}::{answer}"

def eval_with_postprocessing(x, y):
    gold_letter, gold_answer = y.split("::")

    ans_score = 0.0
    try:
        ans = get_boxed_answer(x)
        ans = extract_text_content(ans).lower()
    except:
        ans = ""
    if ans.lower() == gold_letter.lower():
        ans_score = 1.0
    elif ans.lower() == gold_answer.lower():
        ans_score = 1.0
    elif ans.lower()[:1] == gold_letter.lower():
        ans_score = 1.0
    return ans_score

class GlobalMMLULiteTask(YevalTask):
    data_path="CohereLabs/Global-MMLU-Lite"
    input_text=input_text
    output_text=output_text
    test_split="test"
    evaluation={"accuracy": eval_with_postprocessing}
    sample_agg_fn={"accuracy": lambda x: x}
    logging=log_logprob

@register_task("global_mmlu_translate")
class JSONGSM8KTrainTask(GlobalMMLULiteTask):
    data_path="json"
    input_text=lambda x: x["input"]
    output_text=lambda x: x["output"]
    test_split="train"
    # preprocessing=highest_loglikelihood
    preprocessing=highest_language_content

@register_task("global_mmlu_ar")
class GlobalMMLU_AR_Task(GlobalMMLULiteTask):
    data_name="ar"

@register_task("global_mmlu_bn")
class GlobalMMLU_BN_Task(GlobalMMLULiteTask):
    data_name="bn"

@register_task("global_mmlu_de")
class GlobalMMLU_DE_Task(GlobalMMLULiteTask):
    data_name="de"

@register_task("global_mmlu_en")
class GlobalMMLU_EN_Task(GlobalMMLULiteTask):
    data_name="en"

@register_task("global_mmlu_es")
class GlobalMMLU_ES_Task(GlobalMMLULiteTask):
    data_name="es"

@register_task("global_mmlu_fr")
class GlobalMMLU_FR_Task(GlobalMMLULiteTask):
    data_name="fr"

@register_task("global_mmlu_hi")
class GlobalMMLU_HI_Task(GlobalMMLULiteTask):
    data_name="hi"

@register_task("global_mmlu_id")
class GlobalMMLU_ID_Task(GlobalMMLULiteTask):
    data_name="id"

@register_task("global_mmlu_it")
class GlobalMMLU_IT_Task(GlobalMMLULiteTask):
    data_name="it"

@register_task("global_mmlu_ja")
class GlobalMMLU_JA_Task(GlobalMMLULiteTask):
    data_name="ja"

@register_task("global_mmlu_ko")
class GlobalMMLU_KO_Task(GlobalMMLULiteTask):
    data_name="ko"

@register_task("global_mmlu_my")
class GlobalMMLU_MY_Task(GlobalMMLULiteTask):
    data_name="my"

@register_task("global_mmlu_pt")
class GlobalMMLU_PT_Task(GlobalMMLULiteTask):
    data_name="pt"

@register_task("global_mmlu_sw")
class GlobalMMLU_SW_Task(GlobalMMLULiteTask):
    data_name="sw"

@register_task("global_mmlu_yo")
class GlobalMMLU_YO_Task(GlobalMMLULiteTask):
    data_name="yo"

@register_task("global_mmlu_zh")
class GlobalMMLU_ZH_Task(GlobalMMLULiteTask):
    data_name="zh"
