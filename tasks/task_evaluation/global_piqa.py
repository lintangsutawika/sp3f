import os
import re
from functools import partial

from yeval.task import register_task, YevalTask
from yeval.log.usage import log_logprob
from yeval.response.math_responses import get_boxed_answer

from lang_boot.utils import (
    extract_text_content,
    highest_loglikelihood
    )

path = os.path.dirname(__file__)

def input_text(x):
    return f"{x['prompt']}\nA) {x['solution0']}\nB) {x['solution1']}\n\n"

def output_text(x):
    label = int(x["label"])
    letter = ["A", "B"][label]
    answer = x["solution" + str(label)]
    return f"{letter}::{answer}"

def eval_with_postprocessing(x, y):
    gold_letter, gold_answer = y.split("::")

    ans_score = 0.0
    ans = get_boxed_answer(x)
    ans = extract_text_content(ans).lower()
    if ans.lower() == gold_letter.lower():
        ans_score = 1.0
    elif ans.lower() == gold_answer.lower():
        ans_score = 1.0
    elif ans.lower()[:1] == gold_letter.lower():
        ans_score = 1.0
    return ans_score

class GlobalPIQATask(YevalTask):
    data_path="mrlbenchmarks/global-piqa-nonparallel"
    input_text=input_text
    output_text=output_text
    test_split="test"
    evaluation={"accuracy": eval_with_postprocessing}
    sample_agg_fn={"accuracy": lambda x: x}
    logging=log_logprob

@register_task("global_piqa_translate")
class JSONGlobalPIQATask(GlobalPIQATask):
    data_path="json"
    input_text=lambda x: x["input"]
    output_text=lambda x: x["output"]
    test_split="train"
    preprocessing=highest_loglikelihood

@register_task("global_piqa_bn")
class GlobalPIQA_BN_Task(GlobalPIQATask):
    data_name="ben_beng"

@register_task("global_piqa_de")
class GlobalPIQA_DE_Task(GlobalPIQATask):
    data_name="deu_latn"

@register_task("global_piqa_en")
class GlobalPIQA_EN_Task(GlobalPIQATask):
    data_name="eng_latn"

@register_task("global_piqa_es")
class GlobalPIQA_ES_Task(GlobalPIQATask):
    data_name="spa_latn"

@register_task("global_piqa_fr")
class GlobalPIQA_FR_Task(GlobalPIQATask):
    data_name="fra_latn"

@register_task("global_piqa_ja")
class GlobalPIQA_JA_Task(GlobalPIQATask):
    data_name="jpn_jpan"

@register_task("global_piqa_ru")
class GlobalPIQA_RU_Task(GlobalPIQATask):
    data_name="rus_cyrl"

@register_task("global_piqa_sw")
class GlobalPIQA_SW_Task(GlobalPIQATask):
    data_name="swh_latn"

@register_task("global_piqa_te")
class GlobalPIQA_TE_Task(GlobalPIQATask):
    data_name="tel_telu"

@register_task("global_piqa_th")
class GlobalPIQA_TH_Task(GlobalPIQATask):
    data_name="tha_thai"

@register_task("global_piqa_zh")
class GlobalPIQA_ZH_Task(GlobalPIQATask):
    data_name="zho_hans"

@register_task("global_piqa_id")
class GlobalPIQA_ID_Task(GlobalPIQATask):
    data_name="ind_latn"
