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

from sp3f.utils import (
    math_eval_with_postprocessing,
    highest_loglikelihood,
    highest_language_content,
    )

path = os.path.dirname(__file__)

class MT_MATH100Task(YevalTask):
    data_path="amphora/MCLM"
    data_name="MT-MATH100"
    output_text=lambda x: x["answer"]
    test_split="test"
    evaluation={"accuracy": math_eval_with_postprocessing}
    sample_agg_fn={"accuracy": lambda x: x}
    logging=log_logprob

@register_task("mt_math100_translate")
class JSONMT_MATH100Task(MT_MATH100Task):
    data_path="json"
    input_text=lambda x: x["input"]
    output_text=lambda x: x["output"]
    test_split="train"
    # preprocessing=highest_loglikelihood
    preprocessing=highest_language_content

# ['ar', 'bn', 'de', 'en', 'fr', 'hi', 'id', 'it', 'ja', 'ko', 'pt', 'ru', 'sw', 'yo', 'te', 'th', 'zh-cn', 'zh-tw']

@register_task("mt_math100_ar")
class MT_MATH100_AR_Task(MT_MATH100Task):
    input_text=lambda x: x["ar"]

@register_task("mt_math100_bn")
class MT_MATH100_BN_Task(MT_MATH100Task):
    input_text=lambda x: x["bn"]

@register_task("mt_math100_de")
class MT_MATH100_DE_Task(MT_MATH100Task):
    input_text=lambda x: x["de"]

@register_task("mt_math100_en")
class MT_MATH100_EN_Task(MT_MATH100Task):
    input_text=lambda x: x["en"]

@register_task("mt_math100_es")
class MT_MATH100_ES_Task(MT_MATH100Task):
    input_text=lambda x: x["es"]

@register_task("mt_math100_fr")
class MT_MATH100_FR_Task(MT_MATH100Task):
    input_text=lambda x: x["fr"]

@register_task("mt_math100_ja")
class MT_MATH100_JA_Task(MT_MATH100Task):
    input_text=lambda x: x["ja"]

@register_task("mt_math100_ru")
class MT_MATH100_RU_Task(MT_MATH100Task):
    input_text=lambda x: x["ru"]

@register_task("mt_math100_sw")
class MT_MATH100_SW_Task(MT_MATH100Task):
    input_text=lambda x: x["sw"]

@register_task("mt_math100_te")
class MT_MATH100_TW_Task(MT_MATH100Task):
    input_text=lambda x: x["te"]

@register_task("mt_math100_th")
class MT_MATH100_TH_Task(MT_MATH100Task):
    input_text=lambda x: x["th"]

@register_task("mt_math100_zh-cn")
class MT_MATH100_ZH_Task(MT_MATH100Task):
    input_text=lambda x: x["zh-cn"]

@register_task("mt_math100_zh-tw")
class MT_MATH100_ZH_TW_Task(MT_MATH100Task):
    input_text=lambda x: x["zh-tw"]

@register_task("mt_math100_id")
class MT_MATH100_ID_Task(MT_MATH100Task):
    input_text=lambda x: x["id"]

@register_task("mt_math100_it")
class MT_MATH100_IT_Task(MT_MATH100Task):
    input_text=lambda x: x["it"]

@register_task("mt_math100_af")
class MT_MATH100_AF_Task(MT_MATH100Task):
    input_text=lambda x: x["af"]

@register_task("mt_math100_sq")
class MT_MATH100_SQ_Task(MT_MATH100Task):
    input_text=lambda x: x["sq"]

@register_task("mt_math100_bg")
class MT_MATH100_BG_Task(MT_MATH100Task):
    input_text=lambda x: x["bg"]

@register_task("mt_math100_ca")
class MT_MATH100_CA_Task(MT_MATH100Task):
    input_text=lambda x: x["ca"]

@register_task("mt_math100_ko")
class MT_MATH100_KO_Task(MT_MATH100Task):
    input_text=lambda x: x["ko"]

@register_task("mt_math100_pt")
class MT_MATH100_PT_Task(MT_MATH100Task):
    input_text=lambda x: x["pt"]

@register_task("mt_math100_hi")
class MT_MATH100_HI_Task(MT_MATH100Task):
    input_text=lambda x: x["hi"]

@register_task("mt_math100_vi")
class MT_MATH100_VI_Task(MT_MATH100Task):
    input_text=lambda x: x["vi"]

@register_task("mt_math100_af")
class MT_MATH100_AF_Task(MT_MATH100Task):
    input_text=lambda x: x["af"]

@register_task("mt_math100_nl")
class MT_MATH100_NL_Task(MT_MATH100Task):
    input_text=lambda x: x["nl"]

@register_task("mt_math100_gu")
class MT_MATH100_GU_Task(MT_MATH100Task):
    input_text=lambda x: x["gu"]

@register_task("mt_math100_ne")
class MT_MATH100_NE_Task(MT_MATH100Task):
    input_text=lambda x: x["ne"]

@register_task("mt_math100_pa")
class MT_MATH100_PA_Task(MT_MATH100Task):
    input_text=lambda x: x["pa"]

@register_task("mt_math100_tr")
class MT_MATH100_TR_Task(MT_MATH100Task):
    input_text=lambda x: x["tr"]

@register_task("mt_math100_tl")
class MT_MATH100_TL_Task(MT_MATH100Task):
    input_text=lambda x: x["tl"]

@register_task("mt_math100_he")
class MT_MATH100_HE_Task(MT_MATH100Task):
    input_text=lambda x: x["he"]



# 'hr'
# 'cs'
# 'da',
# 'nl'
# 'et'
# 'fi'
# 'el'
# 'gu'
# 'he'
# 'hi'
# 'hu'
# 'id',
# 'it',
# 'kn',
# 'ko',
# 'lv',
# 'lt',
# 'mk',
# 'ml',
# 'mr',
# 'ne',
# 'no',
# 'fa',
# 'pl'
# 'pa'
# 'ro'
# 'ru'
# 'sk'
# 'sl'
# 'so'
# 'sv'
# 'tl',
# 'ta'
# 'tr'
# 'uk'
# 'ur'
# 'vi'
# 'cy'
