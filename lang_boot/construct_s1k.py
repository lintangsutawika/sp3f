import os
import argparse
import pandas as pd

from datasets import load_dataset

from yeval.task import TASK_LIST
from yeval.utils import import_modules

path = os.path.dirname(__file__)
import_modules(os.path.join(path, "../tasks/"))

def from_key(x, key):
    return x[key] if key in x else None


def select_best_candidate(row, col_name="input_candidates", use_logprob=True, use_accuracy=False, use_lang=False):
    
    candidates_dict = {'candidate': row[col_name]}
    
    # Sort by selected criteria (descending order)
    sort_columns = []
    if use_logprob:
        sort_columns.append('logprob')
        candidates_dict["logprob"] = row["logprob"]
    else:
        candidates_dict["logprob"] = [1.0] * len(row[col_name])

    if use_lang:
        sort_columns.append('lang')
        candidates_dict["lang"] = row["lang"]
    else:
        candidates_dict["lang"] = [1.0] * len(row[col_name])

    if use_accuracy:
        sort_columns.append('accuracy')
        candidates_dict["accuracy"] = row["accuracy"]
    else:
        candidates_dict["accuracy"] = [1.0] * len(row[col_name])
    
    candidates_df = pd.DataFrame(candidates_dict)

    candidates_df["score"] = (-1/candidates_df["logprob"]) * candidates_df["lang"] * candidates_df["accuracy"]

    candidates_df = candidates_df.sort_values(by="score", ascending=False)

    if len(candidates_df[candidates_df["score"] > 0]) == 0:
        return "None"
    
    return candidates_df.iloc[0]['candidate']


def construct_dataframe(
    data_path,
    lang="en",
    task="s1k_train",
    ):
    
    df = pd.DataFrame()
    # prompt_fn = TASK_LIST[f"{lang}_reason"].user_message
    prompt_fn = lambda x: x
    system_message = [{"role": "system", "content": TASK_LIST[f"{lang}_system"].system_message}]

    query_path = os.path.join(data_path, f"raw_traces/{task}+{lang}+translated+queries/")
    query_df = pd.read_json(os.path.join(query_path, "output.jsonl"), lines=True)

    query_df['input_candidates'] = query_df.apply(lambda row: from_key(row, "answer"), axis=1)
    query_df['problem'] = query_df.apply(
        lambda row: select_best_candidate(
            row, 
            col_name="input_candidates",
            use_logprob=True,
            use_accuracy=False,
            use_lang=True,
        ), 
        axis=1
    )
    query_df["answer"] = query_df["ground_truth"]

    solution_path = os.path.join(data_path, f"raw_traces/{task}+{lang}+translated+solutions/")
    solution_df = pd.read_json(os.path.join(solution_path, "output.jsonl"), lines=True)

    solution_df['solution_candidates'] = solution_df.apply(lambda row: from_key(row, "answer"), axis=1)
    solution_df['solution'] = solution_df.apply(
        lambda row: select_best_candidate(
            row, 
            col_name="solution_candidates",
            use_logprob=True,
            use_accuracy=False,
            use_lang=True,
        ), 
        axis=1
    )
    query_df['solution'] = solution_df['solution']

    df['solution'] = query_df['solution']

    df['reward_model'] = query_df.apply(
        lambda row: {
            "ground_truth": str(row["answer"])
        },
        axis=1
    )

    df['input'] = query_df.apply(lambda row: system_message + [{"role": "user", "content": prompt_fn(row["problem"])}], axis=1)
    df['output'] = df['solution']
    df['data_source'] = "train_dataset"
    df['raw_prompt'] = query_df.apply(
        lambda row: row["problem"],
        axis=1
    )

    df["extra_info"] = df.apply(
        lambda row: {
            "task": "train_dataset",
            "ground_truth": str(row["reward_model"]["ground_truth"]),
            "use_lang": False,
            "use_accuracy": True,
            "lang": lang,
        }, 
        axis=1
    )

    df["messages"] = df.apply(
        lambda row: row["input"] + [{"role": "assistant", "content": row["output"]}],
        axis=1
    )

    train_df = df

    task_list = ["math500", "mgsm", "global_mmlu", "belebele", "mt_math100"]
    test_df = pd.DataFrame()
    for task_name in task_list:
        if task_name == "math500":
            full_task_name = task_name
            task_prompt_fn = TASK_LIST["en_reason"].user_message
        else:
            full_task_name = f'{task_name}_{lang}'
            task_prompt_fn = prompt_fn

        eval_df = pd.DataFrame()
        eval_dataset = TASK_LIST[full_task_name]()
        x, y = zip(*[eval_dataset.dataset.__getitem__(idx) for idx in range(eval_dataset.dataset.__len__())])

        eval_df['output'] = [str(_y) for _y in y]
        eval_df['raw_prompt'] = x
        eval_df['input'] = eval_df.apply(lambda row: system_message + [{"role": "user", "content": task_prompt_fn(row["raw_prompt"])}], axis=1)
        eval_df['data_source'] = task_name
        eval_df["extra_info"] = eval_df.apply(
            lambda row: {
                "task": task_name,
                "ground_truth": str(row["output"]),
                "lang": lang,
            }, 
            axis=1
        )
        eval_df['reward_model'] = eval_df.apply(
            lambda row: {
                "ground_truth": str(row["output"]),
            },
            axis=1
        )

        eval_df["messages"] = df.apply(
            lambda row: row["input"] + [{"role": "assistant", "content": row["output"]}],
            axis=1
        )

        eval_df = eval_df[['input', 'data_source', 'raw_prompt', 'reward_model', 'extra_info', 'messages']]
        test_df = pd.concat([test_df, eval_df], ignore_index=True)

    output_path = os.path.join(data_path, f"prep_traces/{task}+{lang}/")
    os.makedirs(output_path, exist_ok=True)
    train_df.to_parquet(os.path.join(output_path, "train.parquet"))
    # valid_df.to_parquet(os.path.join(output_path, "valid.parquet"))
    test_df.to_parquet(os.path.join(output_path, "test.parquet"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--lang', type=str, default="en")
    parser.add_argument('--use_solution', action='store_true', default=False, help="Use translated queries")
    args = parser.parse_args()
    construct_dataframe(
        data_path=args.data_path,
        lang=args.lang,
        # use_solution=args.use_solution,
    )
