import os
import re
from functools import partial

from yeval.task import register_task, YevalTask

from yeval.log.usage import log_token_usage

def input_text(x, lang="Indonesian"):
    _x = x[lang]
    _input = [_x['question'], "\n".join(_x['choices'])]
    if 'context' in _x:
        _input.insert(0, _x['context'])   
    return "\n".join(_input)

def output_text(x, lang="Indonesian"):
    _x = x[lang]
    answer = _x['answer']
    letter, *text = answer.split(" ")
    text = " ".join(text)
    return [letter, text, answer]

# class CrossMMLUTask(YevalTask):
#     data_path="SeaEval/cross_mmlu"
#     test_split="test"
#     evaluation={"accuracy": eval}

# @register_task("flores_id_to_en")
# class CrossMMLUIndTask(CrossMMLUTask):
#     input_text=lambda x: partial(input_text, lang="Indonesian")(x)
#     output_text=lambda x: partial(output_text, lang="Indonesian")(x)

