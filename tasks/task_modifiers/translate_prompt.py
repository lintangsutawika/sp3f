import re

from functools import partial

from yeval.task import register_task, YevalTask
from yeval.log.usage import log_logprob

from sp3f.utils import get_lang_score

def lang_content(x, y, lang="id"):
    _, lang_prob = get_lang_score(x, lang=lang)
    return lang_prob

def get_translation(x):
    try:
        pattern = rf"```(.*?)```"
        match = re.search(pattern, x, re.DOTALL)
        if match:
            return match.group(1).strip()
        elif "English Translation:" in x:
            return x.split("English Translation:")[-1].strip()
        # return ""
        else:
            return x
    except:
        return x

class BaseTranslateTask(YevalTask):
    postprocessor=get_translation
    sampling_args={
        "n": 4,
        "temperature": 1.0,
        # "logprobs": True,
        # "stop": ["```"],
        # "extra_body": {"include_stop_str_in_output": True}
        }
    sample_agg_fn={"lang": lambda x: x}
    # logging=log_logprob

@register_task("ar_translate")
class ARTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Arabic. \
Respond directly after \"Arabic Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"{x}"+"Arabic Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="ar")}

@register_task("bn_translate")
class BNTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Bengali. \
Respond directly after \"Bengali Translation:\".\
"""
    # user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"
    user_message=lambda x: "English Text:\n\n"+f"{x}"+"Bengali Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="bn")}

@register_task("de_translate")
class DETranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to German. \
Respond directly after \"German Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"{x}"+"German Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="de")}

@register_task("en_translate")
class ENTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate to English. \
Respond directly after \"English Translation:\".\
"""
    user_message=lambda x: "Source Text:\n\n"+f"{x}"+"English Translation:\n\n"
    evaluation={
        "lang": partial(lang_content, lang="en"),
        "accuracy": lambda x, y: -1,
    }

@register_task("it_translate")
class ITTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Italian. \
Respond directly after \"Italian Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"{x}"+"Italian Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="it")}

@register_task("ko_translate")
class KOTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Korean. \
Respond directly after \"Korean Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"{x}"+"Korean Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="ko")}

@register_task("pt_translate")
class PTTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Portuguese. \
Respond directly after \"Portuguese Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"{x}"+"Portuguese Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="pt")}

@register_task("yo_translate")
class YOTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Yoruba. \
Respond directly after \"Yoruba Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"{x}"+"Yoruba Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="yo")}

@register_task("hi_translate")
class HITranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Hindi. \
Respond directly after \"Hindi Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"{x}"+"Hindi Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="hi")}


@register_task("es_translate")
class ESTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Spanish. \
Respond directly after \"Spanish Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"{x}"+"Spanish Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="es")}

@register_task("fr_translate")
class FRTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to French. \
Respond directly after \"French Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"{x}"+"French Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="fr")}

@register_task("ja_translate")
class JATranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Japanese. \
Respond directly after \"Japanese Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"{x}"+"Japanese Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="ja")}

@register_task("ru_translate")
class RUTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Russian. \
Respond directly after \"Russian Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"{x}"+"Russian Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="ru")}

@register_task("sw_translate")
class SWTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Swahili. \
Respond directly after \"Swahili Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"{x}"+"Swahili Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="sw")}

@register_task("te_translate")
class TWTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Telugu. \
Respond directly after \"Telugu Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"{x}"+"Telugu Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="te")}

@register_task("th_translate")
class THTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Thai. \
Respond directly after \"Thai Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"{x}"+"Thai Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="th")}

@register_task("zh_translate")
class JPNTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Chinese. \
Respond directly after \"Chinese Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"{x}"+"Chinese Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="zh")}

@register_task("id_translate")
class IDTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Indonesian. \
Respond directly after \"Indonesian Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"{x}"+"Indonesian Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="id")}

@register_task("default")
class TranslateTask(YevalTask):
    input_text=lambda x: x["answer"][0]
    output_text=lambda x: x["ground_truth"]
    data_path="json"
    sampling_args={
        "n": 4,
    }
    test_split="train"
