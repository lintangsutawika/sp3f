import os
import random

import numpy as np
from functools import partial
from yeval.task import register_task, YevalTask

from lang_boot.utils import get_boxed_answer

path = os.path.dirname(__file__)

def get_answer(x):
    return get_boxed_answer("\\boxed{" + x)

def choice_mapping(dataset, correct_is_A=True, use_random=False):

    def map_fn(example, correct_is_A=True, use_random=False):

        if use_random:
            if random.random() > 0.5:
                correct_is_A = True
            else:
                correct_is_A = False

        if correct_is_A:
            example["A"] = example["yes"]
            example["B"] = example["no"]
            example["label"] = "A"
        else:
            example["A"] = example["no"]
            example["B"] = example["yes"]
            example["label"] = "B"
        return example

    _fn = partial(map_fn, correct_is_A=correct_is_A, use_random=use_random)
    dataset = dataset.map(_fn)
    return dataset

class JGTask(YevalTask):
    evaluation={"accuracy": lambda x, y: int(x == y)}
    # evaluation={"accuracy": lambda x, y: int(get_answer(x) == y)}
    # evaluation={"accuracy": lambda x, y: int(get_boxed_answer(x) == y)}
    sample_agg_fn={"accuracy": lambda x: np.mean(x)}
    output_text=lambda x: x["label"]
    sampling_args={
        "stop": ["}"],
        "extra_body": {
            # "include_stop_str_in_output": True,
            "guided_choice": ["A", "B"],
        },
    }
    test_split="test"
    data_path="json"
    data_name=None
    input_text=lambda x: f"""English Response START
{x["original"]}
English Response END

Response A START
{x["A"]}
Response A END

Response B START
{x["B"]}
Response B END

Using the following considerations:
1. The response MUST be fluent and coherent in Indonesian.
2  The response MUST NOT be in English. 
3. The response MUST be a suitable translation of the English text.

Between response \\boxed{{A}} or response \\boxed{{B}} the best response according to the listed considerations """ + "is \\boxed{"
# Query START
# {x["query"]}
# Query END
# Between response \\boxed{{A}} or response \\boxed{{B}} which is better? Write your answer in \\boxed{{}} /no_think"""
# 3. The response that is the most accurate
# 4. The response that is the most helpful for the user.

# learning towards g_right
@register_task("judge_grounding_1A")
class JG1ATask(JGTask):
    data_kwargs={"data_files": {"test": os.path.join(path, "g_right_v_g_wrong.jsonl")}}
    preprocessing=partial(choice_mapping, correct_is_A=True)

@register_task("judge_grounding_1B")
class JG1BTask(JG1ATask):
    preprocessing=partial(choice_mapping, correct_is_A=False)

@register_task("judge_grounding_1C")
class JG1CTask(JG1ATask):
    preprocessing=partial(choice_mapping, use_random=True)

# Slightly learning towards t
@register_task("judge_grounding_2A")
class JG2ATask(JGTask):
    data_kwargs={"data_files": {"test": os.path.join(path, "g_right_v_t.jsonl")}}
    preprocessing=partial(choice_mapping, correct_is_A=True)

@register_task("judge_grounding_2B")
class JG2BTask(JG2ATask):
    preprocessing=partial(choice_mapping, correct_is_A=False)

@register_task("judge_grounding_2C")
class JG2CTask(JG2ATask):
    preprocessing=partial(choice_mapping, use_random=True)

# Slightly learning towards t
@register_task("judge_grounding_3A")
class JG3ATask(JGTask):
    data_kwargs={"data_files": {"test": os.path.join(path, "g_wrong_v_t.jsonl")}}
    preprocessing=partial(choice_mapping, correct_is_A=True)

@register_task("judge_grounding_3B")
class JG3BTask(JG3ATask):
    preprocessing=partial(choice_mapping, correct_is_A=False)

@register_task("judge_grounding_3C")
class JG3CTask(JG3ATask):
    preprocessing=partial(choice_mapping, use_random=True)

# Slightly learning towards t
@register_task("judge_grounding_4A")
class JG2ATask(JGTask):
    data_kwargs={"data_files": {"test": os.path.join(path, "g_right_v_en_right.jsonl")}}
    preprocessing=partial(choice_mapping, correct_is_A=True)

@register_task("judge_grounding_4B")
class JG2BTask(JG2ATask):
    preprocessing=partial(choice_mapping, correct_is_A=False)

@register_task("judge_grounding_4C")
class JG2CTask(JG2ATask):
    preprocessing=partial(choice_mapping, use_random=True)

# Slightly learning towards t
@register_task("judge_grounding_5A")
class JG3ATask(JGTask):
    data_kwargs={"data_files": {"test": os.path.join(path, "t_v_en_right.jsonl")}}
    preprocessing=partial(choice_mapping, correct_is_A=True)

@register_task("judge_grounding_5B")
class JG3BTask(JG3ATask):
    preprocessing=partial(choice_mapping, correct_is_A=False)

@register_task("judge_grounding_5C")
class JG3CTask(JG3ATask):
    preprocessing=partial(choice_mapping, use_random=True)
