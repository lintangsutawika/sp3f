import re
import random
import numpy as np

from functools import partial

from yeval.task import register_task, YevalTask
from yeval.log.usage import log_logprob

from lang_boot.utils import (
    highest_loglikelihood,
    highest_language_content,
    math_eval_with_postprocessing,
)

def shuffle(dataset, seed=0):
    shuffled_dataset = dataset.shuffle(seed=seed)
    return shuffled_dataset.flatten_indices()

@register_task("s1k_train")
class LangBootS1KTrainTask(YevalTask):
    data_path="simplescaling/s1K-1.1"
    test_split="train"
    input_text=lambda x: x["question"]
    output_text=lambda x: x["solution"]
    postprocessor=None
    evaluation={"accuracy": math_eval_with_postprocessing}
    sample_agg_fn={"accuracy": lambda x: x}
    logging=log_logprob

@register_task("json_highest_log_s1k_train")
class JSONS1KTrainTask(LangBootS1KTrainTask):
    data_path="json"
    input_text=lambda x: x["input"]
    output_text=lambda x: x["output"]
    preprocessing=highest_loglikelihood

@register_task("json_highest_lang_s1k_train")
class JSONS1KTrainTask(LangBootS1KTrainTask):
    data_path="json"
    input_text=lambda x: x["input"].split("Translation:")[-1].strip()
    output_text=lambda x: x["output"]
    preprocessing=highest_language_content

@register_task("s1k_train_problem")
class ProblemS1KTrainTask(LangBootS1KTrainTask):
    input_text=lambda x: x["question"]
    evaluation=None

@register_task("s1k_train_solution")
class OpenR1Math220KTask(LangBootS1KTrainTask):
    # input_text=lambda x: "<t>" + x["gemini_thinking_trajectory"] + "<\\t>" + x["gemini_attempt"]
    input_text=lambda x: x["gemini_attempt"]
    evaluation=None

@register_task("s1k_test")
class LangBootS1KTestTask(LangBootS1KTrainTask):
    test_split="test"
    # postprocessor=get_boxed_answer
    postprocessor=None
    evaluation={"accuracy": math_eval_with_postprocessing}

@register_task("s1k_test_problem")
class OpenR1Math220KTask(LangBootS1KTrainTask):
    input_text=lambda x: x["problem"]
    evaluation=None

@register_task("s1k_test_solution")
class OpenR1Math220KTask(LangBootS1KTrainTask):
    input_text=lambda x: x["answer"]
    evaluation=None

if __name__ == "__main__":
    pass
