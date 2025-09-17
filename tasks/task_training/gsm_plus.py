import re
import random
import numpy as np

from functools import partial

from yeval.task import register_task, YevalTask
from yeval.task.gsm8k import GSM8KTask
from yeval.log.usage import log_logprob

from lang_boot.utils import (
    highest_loglikelihood,
    highest_language_content,
    math_eval_with_postprocessing,
)

def shuffle(dataset, seed=0):
    shuffled_dataset = dataset.shuffle(seed=seed)
    return shuffled_dataset.flatten_indices()

@register_task("gsm_plus_train")
class LangBootGSMPlusTrainTask(GSM8KTask):
    data_path="qintongli/GSM-Plus"
    data_name=None
    test_split="test"
    input_text=lambda x: x["question"]
    output_text=lambda x: x["answer"]
    postprocessor=None
    evaluation={"accuracy": math_eval_with_postprocessing}
    sample_agg_fn={"accuracy": lambda x: x}
    logging=log_logprob

@register_task("json_highest_log_gsm_plus_train")
class JSONGSMPlusTrainTask(LangBootGSMPlusTrainTask):
    data_path="json"
    input_text=lambda x: x["input"]
    output_text=lambda x: x["output"]
    preprocessing=highest_loglikelihood

@register_task("json_highest_lang_gsm_plus_train")
class JSONGSMPlusTrainTask(LangBootGSMPlusTrainTask):
    data_path="json"
    input_text=lambda x: x["input"]
    output_text=lambda x: x["output"]
    preprocessing=highest_language_content

@register_task("gsm_plus_train_problem")
class ProblemGSMPlusTrainTask(LangBootGSMPlusTrainTask):
    input_text=lambda x: x["question"]
    evaluation=None

@register_task("gsm_plus_train_solution")
class OpenR1Math220KTask(LangBootGSMPlusTrainTask):
    input_text=lambda x: x["answer"]
    evaluation=None

# @register_task("gsm_plus_test")
# class LangBootGSMPlusTestTask(GSMPlusTask):
#     test_split="test"
#     # postprocessor=get_boxed_answer
#     postprocessor=None
#     evaluation={"accuracy": math_eval_with_postprocessing}

# @register_task("gsm_plus_test_problem")
# class OpenR1Math220KTask(LangBootGSMPlusTrainTask):
#     input_text=lambda x: x["question"]
#     evaluation=None

# @register_task("gsm_plus_test_solution")
# class OpenR1Math220KTask(LangBootGSMPlusTrainTask):
#     input_text=lambda x: x["answer"]
#     evaluation=None

if __name__ == "__main__":
    pass
