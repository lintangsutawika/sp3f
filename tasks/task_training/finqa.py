import re
import random
import numpy as np

from functools import partial

from yeval.task import register_task, YevalTask
from yeval.log.usage import log_logprob
from yeval.response.math_responses import get_boxed_answer

from lang_boot.utils import (
    highest_loglikelihood,
    highest_language_content,
    math_eval_with_postprocessing,
)

def finqa_input(x):

    pre_text = "\n".join(x['pre_text'])
    post_text = "\n".join(x['post_text'])
    table = "\n".join([" | ".join(line) for line in x['table']])
    question = x['question']

    return f"{pre_text}\n\nTable:\n{table}\n\n{post_text}\n{question}\nAnswer"

def finqa_output(x):
    return x['answer']

def match_decimals(prediction, ground_truth):
    reversed_number = str(ground_truth)[::-1]
    decimal_places = reversed_number.find('.')
    decimal_places = decimal_places if decimal_places != -1 else 0
    if decimal_places == 1 and reversed_number[0] == "0":
        decimal_places = 0
    rounded_prediction = round(prediction, decimal_places)
    return rounded_prediction


def finqa_eval(prediction, ground_truth):

    prediction = get_boxed_answer(prediction)
    try:

        if ground_truth in ["yes", "no"]:
            if prediction == "True":
                prediction = "yes"
            elif prediction == "False":
                prediction = "no"

            score = 1 if prediction == ground_truth else 0

        else:
            prediction = float(prediction)
            if "%" in ground_truth:
                ground_truth = ground_truth.replace("%", "")
                if prediction < 1.0:
                    prediction = prediction * 100    
                
            ground_truth = float(ground_truth)
            prediction = match_decimals(prediction, ground_truth)
            score = 1 if abs(prediction - ground_truth) < 1e-3 else 0

    except Exception as e:
        prediction = str(prediction)
        ground_truth = str(ground_truth)
        score = 1 if prediction == ground_truth else 0

    return score

def shuffle(dataset, seed=0):
    shuffled_dataset = dataset.shuffle(seed=seed)
    return shuffled_dataset.flatten_indices()

@register_task("finqa_train")
class LangBootFinQATrainTask(YevalTask):
    data_path="ibm-research/finqa"
    test_split="train"
    input_text=finqa_input
    output_text=finqa_output
    evaluation={"accuracy": finqa_eval}
    sample_agg_fn={"accuracy": lambda x: x}
    logging=log_logprob

@register_task("json_highest_log_finqa_train")
class JSONFinQATrainTask(LangBootFinQATrainTask):
    data_path="json"
    input_text=lambda x: x["input"]
    output_text=lambda x: x["output"]
    preprocessing=highest_loglikelihood

@register_task("json_highest_lang_finqa_train")
class JSONFinQATrainTask(LangBootFinQATrainTask):
    data_path="json"
    input_text=lambda x: x["input"]
    output_text=lambda x: x["output"]
    preprocessing=highest_language_content

@register_task("finqa_train_problem")
class ProblemFinQATrainTask(LangBootFinQATrainTask):
    input_text=finqa_input
    evaluation=None

@register_task("finqa_train_solution")
class OpenR1Math220KTask(LangBootFinQATrainTask):
    input_text=lambda x: x["answer"]
    evaluation=None

# @register_task("finqa_test")
# class LangBootFinQATestTask(FinQATask):
#     test_split="test"
#     # postprocessor=get_boxed_answer
#     postprocessor=None
#     evaluation={"accuracy": math_eval_with_postprocessing}

# @register_task("finqa_test_problem")
# class OpenR1Math220KTask(LangBootFinQATrainTask):
#     input_text=lambda x: x["question"]
#     evaluation=None

# @register_task("finqa_test_solution")
# class OpenR1Math220KTask(LangBootFinQATrainTask):
#     input_text=lambda x: x["answer"]
#     evaluation=None

if __name__ == "__main__":
    pass
