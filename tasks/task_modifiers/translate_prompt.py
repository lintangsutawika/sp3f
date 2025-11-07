import re

from functools import partial

from yeval.task import register_task, YevalTask
from yeval.log.usage import log_logprob

from lang_boot.utils import get_lang_score

def lang_content(x, y, lang="id"):
    _, lang_prob = get_lang_score(x, lang=lang)
    return lang_prob

def get_translation(x):
    pattern = rf"```(.*?)```"
    try:
        match = re.search(pattern, x, re.DOTALL)
        if match:
            return match.group(1).strip()
        elif "Translation:" in x:
            return x.split("Translation:")[-1].strip()
        else:
            return x
    except Exception as e:
        return str(x)

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
    logging=log_logprob

@register_task("yo_translate")
class YOTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Yoruba. \
Respond directly after \"Yoruba Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"+"\n\nYoruba Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="yo")}

@register_task("pt_translate")
class PTTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Portuguese. \
Respond directly after \"Portuguese Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"+"\n\nPortuguese Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="pt")}

@register_task("my_translate")
class MYTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Burmese. \
Respond directly after \"Burmese Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"+"\n\nBurmese Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="my")}

@register_task("ko_translate")
class KOTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Korean. \
Respond directly after \"Korean Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"+"\n\nKorean Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="ko")}

@register_task("it_translate")
class ITTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Italian. \
Respond directly after \"Italian Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"+"\n\nItalian Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="it")}

@register_task("hi_translate")
class HITranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Hindi. \
Respond directly after \"Hindi Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"+"\n\nHindi Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="hi")}

@register_task("bn_translate")
class BNTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Bengali. \
Respond directly after \"Bengali Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"+"\n\nBengali Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="bn")}

@register_task("de_translate")
class DETranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to German. \
Respond directly after \"German Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"+"\n\nGerman Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="de")}

@register_task("en_translate")
class ENTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate to English. \
Respond directly after \"English Translation:\".\
"""
    user_message=lambda x: "Source Text:\n\n"+f"```\n{x}\n```"+"\n\nEnglish Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="en")}

@register_task("es_translate")
class ESTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Spanish. \
Respond directly after \"Spanish Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"+"\n\nSpanish Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="es")}

@register_task("fr_translate")
class FRTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to French. \
Respond directly after \"French Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"+"\n\nFrench Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="fr")}

@register_task("ja_translate")
class JATranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Japanese. \
Respond directly after \"Japanese Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"+"\n\nJapanese Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="ja")}

@register_task("ru_translate")
class RUTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Russian. \
Respond directly after \"Russian Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"+"\n\nRussian Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="ru")}

@register_task("sw_translate")
class SWTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Swahili. \
Respond directly after \"Swahili Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"+"\n\nSwahili Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="sw")}

@register_task("te_translate")
class TWTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Telugu. \
Respond directly after \"Telugu Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"+"\n\nTelugu Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="te")}

@register_task("th_translate")
class THTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Thai. \
Respond directly after \"Thai Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"+"\n\nThai Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="th")}

@register_task("zh_translate")
class JPNTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Chinese. \
Respond directly after \"Chinese Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"+"\n\nChinese Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="zh")}

@register_task("id_translate")
class IDTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Indonesian. \
Respond directly after \"Indonesian Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"+"\n\nIndonesian Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="id")}

@register_task("ar_translate")
class ARTranslateTask(BaseTranslateTask):
    system_message="""\
You are a helpful assistant that can translate from English to Arabic. \
Respond directly after \"Arabic Translation:\".\
"""
    user_message=lambda x: "English Text:\n\n"+f"```\n{x}\n```"+"\n\nArabic Translation:\n\n"
    evaluation={"lang": partial(lang_content, lang="ar")}

@register_task("default")
class TranslateTask(YevalTask):
    input_text=lambda x: x["answer"][0]
    output_text=lambda x: x["ground_truth"]
    data_path="json"
    sampling_args={
        "n": 4,
    }
    test_split="train"
