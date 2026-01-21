import os
import re
from functools import partial

from yeval.task import register_task, YevalTask
from yeval.log.usage import log_logprob
from yeval.response.math_responses import get_boxed_answer

from lang_boot.utils import (
    extract_text_content,
    highest_loglikelihood,
    highest_language_content,
    )

path = os.path.dirname(__file__)

def input_text(x):
    return f"{x['flores_passage']}\n{x['question']}\n\nA) {x['mc_answer1']}\nB) {x['mc_answer2']}\nC) {x['mc_answer3']}\nD) {x['mc_answer4']}\n\n"

def output_text(x):
    label = int(x["correct_answer_num"])
    letter = ["A", "B", "C", "D"][label - 1]
    answer = x[f"mc_answer{label}"]
    return f"{letter}::{answer}"

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

class BelebeleTask(YevalTask):
    data_path="facebook/belebele"
    input_text=input_text
    output_text=output_text
    test_split="test"
    evaluation={"accuracy": eval_with_postprocessing}
    sample_agg_fn={"accuracy": lambda x: x}
    logging=log_logprob

@register_task("belebele_translate")
class JSONBelebeleTask(BelebeleTask):
    data_path="json"
    input_text=lambda x: x["input"]
    output_text=lambda x: x["output"]
    test_split="train"
    # preprocessing=highest_loglikelihood
    preprocessing=highest_language_content

@register_task("belebele_ar")
class Belebele_AR_Task(BelebeleTask):
    data_name="arb_Arab"

@register_task("belebele_bn")
class Belebele_BN_Task(BelebeleTask):
    data_name="ben_Beng"

@register_task("belebele_de")
class Belebele_DE_Task(BelebeleTask):
    data_name="deu_Latn"

@register_task("belebele_en")
class Belebele_EN_Task(BelebeleTask):
    data_name="eng_Latn"

@register_task("belebele_es")
class Belebele_ES_Task(BelebeleTask):
    data_name="spa_Latn"

@register_task("belebele_hi")
class Belebele_HI_Task(BelebeleTask):
    data_name="hin_Deva"

@register_task("belebele_id")
class Belebele_ID_Task(BelebeleTask):
    data_name="ind_Latn"

@register_task("belebele_it")
class Belebele_IT_Task(BelebeleTask):
    data_name="ita_Latn"

@register_task("belebele_fr")
class Belebele_FR_Task(BelebeleTask):
    data_name="fra_Latn"

@register_task("belebele_ja")
class Belebele_JA_Task(BelebeleTask):
    data_name="jpn_Jpan"

@register_task("belebele_ko")
class Belebele_KO_Task(BelebeleTask):
    data_name="kor_Hang"

@register_task("belebele_pt")
class Belebele_PT_Task(BelebeleTask):
    data_name="por_Latn"

@register_task("belebele_ru")
class Belebele_RU_Task(BelebeleTask):
    data_name="rus_Cyrl"

@register_task("belebele_sw")
class Belebele_SW_Task(BelebeleTask):
    data_name="swh_Latn"

@register_task("belebele_te")
class Belebele_TE_Task(BelebeleTask):
    data_name="tel_Telu"

@register_task("belebele_th")
class Belebele_TH_Task(BelebeleTask):
    data_name="tha_Thai"

@register_task("belebele_zh")
class Belebele_ZH_Task(BelebeleTask):
    data_name="zho_Hans"

@register_task("belebele_yo")
class Belebele_YO_Task(BelebeleTask):
    data_name="yor_Latn"

@register_task("belebele_my")
class Belebele_MY_Task(BelebeleTask):
    data_name="msa_Latn"

@register_task("belebele_vi")
class Belebele_VI_Task(BelebeleTask):
    data_name="vie_Latn"

@register_task("belebele_af")
class Belebele_AF_Task(BelebeleTask):
    data_name="afr_Latn"

@register_task("belebele_nl")
class Belebele_NL_Task(BelebeleTask):
    data_name="nld_Latn"

@register_task("belebele_gu")
class Belebele_GU_Task(BelebeleTask):
    data_name="guj_Gujr"

@register_task("belebele_ne")
class Belebele_NE_Task(BelebeleTask):
    data_name="npi_Deva"

@register_task("belebele_pa")
class Belebele_PA_Task(BelebeleTask):
    data_name="pan_Guru"

@register_task("belebele_tr")
class Belebele_TR_Task(BelebeleTask):
    data_name="tur_Latn"

@register_task("belebele_tl")
class Belebele_TL_Task(BelebeleTask):
    data_name="tgl_Latn"

@register_task("belebele_he")
class Belebele_HE_Task(BelebeleTask):
    data_name="heb_Hebr"