import re
import random
import numpy as np

from functools import partial

from yeval.task import register_task, YevalTask
from yeval.log.usage import log_logprob

from sp3f.utils import (
    highest_loglikelihood,
    highest_language_content,
    math_eval_with_postprocessing,
)

def shuffle(dataset, seed=0):
    shuffled_dataset = dataset.shuffle(seed=seed)
    return shuffled_dataset.flatten_indices()

@register_task("deepscaler_train")
class LangBootDeepScaleRTrainTask(YevalTask):
    data_path="agentica-org/DeepScaleR-Preview-Dataset"
    test_split="train"
    input_text=lambda x: x["problem"]
    output_text=lambda x: x["answer"]
    postprocessor=None
    evaluation={"accuracy": math_eval_with_postprocessing}
    sample_agg_fn={"accuracy": lambda x: x}
    logging=log_logprob

@register_task("json_highest_log_deepscaler_train")
class JSONDeepScaleRTrainTask(LangBootDeepScaleRTrainTask):
    data_path="json"
    input_text=lambda x: x["input"]
    output_text=lambda x: x["output"]
    preprocessing=highest_loglikelihood

@register_task("json_highest_lang_deepscaler_train")
class JSONDeepScaleRTrainTask(LangBootDeepScaleRTrainTask):
    data_path="json"
    input_text=lambda x: x["input"].split("Translation:")[-1].strip()
    output_text=lambda x: x["output"]
    preprocessing=highest_language_content

@register_task("deepscaler_train_problem")
class ProblemDeepScaleRTrainTask(LangBootDeepScaleRTrainTask):
    input_text=lambda x: x["problem"]
    evaluation=None

@register_task("deepscaler_train_solution")
class OpenR1Math220KTask(LangBootDeepScaleRTrainTask):
    input_text=lambda x: x["solution"]
    evaluation=None

@register_task("deepscaler_test")
class LangBootDeepScaleRTestTask(LangBootDeepScaleRTrainTask):
    test_split="test"
    # postprocessor=get_boxed_answer
    postprocessor=None
    evaluation={"accuracy": math_eval_with_postprocessing}

@register_task("parquet_deepscaler_train_problem")
class ParquetDeepScaleRTrainTask(LangBootDeepScaleRTrainTask):
    data_path="parquet"
    input_text=lambda x: x['input'][-1]['content']
    output_text=lambda x: x['extra_info']['ground_truth']
    aux_keys=["solution"]

@register_task("deepscaler_test_problem")
class OpenR1Math220KTask(LangBootDeepScaleRTrainTask):
    input_text=lambda x: x["problem"]
    evaluation=None

@register_task("deepscaler_test_solution")
class OpenR1Math220KTask(LangBootDeepScaleRTrainTask):
    input_text=lambda x: x["answer"]
    evaluation=None

if __name__ == "__main__":
    pass
