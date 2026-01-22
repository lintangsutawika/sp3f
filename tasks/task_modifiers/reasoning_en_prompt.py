from functools import partial

from yeval.response.math_responses import get_boxed_answer
from yeval.task import register_task, YevalTask
from yeval.log.usage import log_logprob

from sp3f.utils import get_lang_score

def lang_content(x, y, lang="id"):
    _, lang_prob = get_lang_score(x, lang=lang)
    return lang_prob

class BaseLangTask(YevalTask):
    sample_agg_fn={"lang": lambda x: x}
    logging=log_logprob


@register_task("en_system-en")
class ENTranslateTask(BaseLangTask):
    system_message="Reason step by step and put your final answer within \\boxed{}."
    evaluation={"lang": partial(lang_content, lang="en")}

@register_task("en_system-ar")
class ARTranslateTask(BaseLangTask):
    system_message="Reason step by step and put your final answer within \\boxed{}."
    evaluation={"lang": partial(lang_content, lang="ar")}

@register_task("en_system-id")
class IDTranslateTask(BaseLangTask):
    system_message="Reason step by step and put your final answer within \\boxed{}."
    evaluation={"lang": partial(lang_content, lang="id")}

@register_task("en_system-ja")
class JATranslateTask(BaseLangTask):
    system_message="Reason step by step and put your final answer within \\boxed{}."
    evaluation={"lang": partial(lang_content, lang="ja")}

@register_task("en_system-bn")
class BNTranslateTask(BaseLangTask):
    system_message="Reason step by step and put your final answer within \\boxed{}."
    evaluation={"lang": partial(lang_content, lang="bn")}

@register_task("en_system-hi")
class HITranslateTask(BaseLangTask):
    system_message="Reason step by step and put your final answer within \\boxed{}."
    evaluation={"lang": partial(lang_content, lang="hi")}

@register_task("en_system-it")
class ITTranslateTask(BaseLangTask):
    system_message="Reason step by step and put your final answer within \\boxed{}."
    evaluation={"lang": partial(lang_content, lang="it")}

@register_task("en_system-ko")
class KOTranslateTask(BaseLangTask):
    system_message="Reason step by step and put your final answer within \\boxed{}."
    evaluation={"lang": partial(lang_content, lang="ko")}

@register_task("en_system-my")
class MYTranslateTask(BaseLangTask):
    system_message="Reason step by step and put your final answer within \\boxed{}."
    evaluation={"lang": partial(lang_content, lang="my")}

@register_task("en_system-pt")
class PTTranslateTask(BaseLangTask):
    system_message="Reason step by step and put your final answer within \\boxed{}."
    evaluation={"lang": partial(lang_content, lang="pt")}

@register_task("en_system-yo")
class YOTranslateTask(BaseLangTask):
    system_message="Reason step by step and put your final answer within \\boxed{}."
    evaluation={"lang": partial(lang_content, lang="yo")}

@register_task("en_system-de")
class DETranslateTask(BaseLangTask):
    system_message="Reason step by step and put your final answer within \\boxed{}."
    evaluation={"lang": partial(lang_content, lang="de")}

@register_task("en_system-es")
class ESTranslateTask(BaseLangTask):
    system_message="Reason step by step and put your final answer within \\boxed{}."
    evaluation={"lang": partial(lang_content, lang="es")}

@register_task("en_system-fr")
class FRTranslateTask(BaseLangTask):
    system_message="Reason step by step and put your final answer within \\boxed{}."
    evaluation={"lang": partial(lang_content, lang="fr")}

@register_task("en_system-ru")
class RUTranslateTask(BaseLangTask):
    system_message="Reason step by step and put your final answer within \\boxed{}."
    evaluation={"lang": partial(lang_content, lang="ru")}

@register_task("en_system-sw")
class SWTranslateTask(BaseLangTask):
    system_message="Reason step by step and put your final answer within \\boxed{}."
    evaluation={"lang": partial(lang_content, lang="sw")}

@register_task("en_system-te")
class TWTranslateTask(BaseLangTask):
    system_message="Reason step by step and put your final answer within \\boxed{}."
    evaluation={"lang": partial(lang_content, lang="te")}

@register_task("en_system-th")
class THTranslateTask(BaseLangTask):
    system_message="Reason step by step and put your final answer within \\boxed{}."
    evaluation={"lang": partial(lang_content, lang="th")}

@register_task("en_system-zh")
class JPNTranslateTask(BaseLangTask):
    system_message="Reason step by step and put your final answer within \\boxed{}."
    evaluation={"lang": partial(lang_content, lang="zh")}
