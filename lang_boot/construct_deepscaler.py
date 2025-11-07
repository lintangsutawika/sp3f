import os
import argparse
import numpy as np
import pandas as pd

from datasets import load_dataset

from yeval.task import TASK_LIST
from yeval.utils import import_modules

from yeval.response.math_responses import get_boxed_answer

path = os.path.dirname(__file__)
import_modules(os.path.join(path, "../tasks/"))

def from_key(x, key):
    return x[key] if key in x else None

lang_answer = {
    "en": f"\n\nThus, the answer is ",
    "ar": f"\n\nلذا، فإن الإجابة هي ",
    "it": f"\n\nQuindi, la risposta è ",
    "fr": f"\n\nDonc, la réponse est ",
    "de": f"\n\nDaher ist die Antwort ",
    "hi": f"\n\nइसलिए, उत्तर है ",
    "ko": f"\n\n따라서, 답은 ",
    "my": f"\n\nထို့ကြောင့်၊ အဖြေမှာ ",
    "pt": f"\n\nPortanto, a resposta é ",
    "ru": f"\n\nТаким образом, ответ ",
    "th": f"\n\nดังนั้น คำตอบคือ ",
    "yo": f"\n\nNítorí náà, ìdáhùn rẹ̀ ni ",
    "zh": f"\n\n因此，答案是 ",
    "te": f"\n\nకాబట్టి, సమాధానం ",
    "th": f"\n\nดังนั้น คำตอบคือ ",
    "id": f"\n\nJadi, jawabannya adalah ",
    "bn": f"\n\nঅতএব, উত্তর হল ",
    "sw": f"\n\nHivyo, jibu ni ",
    "ja": f"\n\nしたがって、答えは ",
    "es": f"\n\nPor lo tanto, la respuesta es ",
}

def fix_answer(x, lang="en", answer_key="answer"):
    solution = x["solution"]
    answer = x[answer_key]
    if solution == "":
        return solution

    ans = get_boxed_answer(solution)
    if ans == "None":
        return solution + lang_answer[lang] + f"\\boxed{{{answer}}}."
        # return solution
    else:
        solution = solution.replace(f"\\boxed{{{ans}}}", f"\\boxed{{{answer}}}")
        return solution

def select_best_candidate(row, col_name="input_candidates", use_logprob=True, use_accuracy=False, use_lang=False, use_parsability=False):
    
    try:
        candidates_dict = {'candidate': row[col_name]}
        
        # Sort by selected criteria (descending order)
        sort_columns = []
        if use_logprob:
            sort_columns.append('logprob')
            candidates_dict["logprob"] = np.exp(np.asarray(row["logprob"]))
        else:
            candidates_dict["logprob"] = ([0.0] * len(row[col_name]))

        if use_lang:
            sort_columns.append('lang')
            candidates_dict["lang"] = row["lang"]
        else:
            candidates_dict["lang"] = [0.0] * len(row[col_name])

        if use_accuracy:
            sort_columns.append('accuracy')
            candidates_dict["accuracy"] = row["accuracy"]
        else:
            candidates_dict["accuracy"] = [0.0] * len(row[col_name])

        if use_parsability:
            sort_columns.append('parsability')
            candidates_dict["parsability"] = [1 if get_boxed_answer(row) != "None" else 0 for row in row[col_name]]
        else:
            candidates_dict["parsability"] = [0.0] * len(row[col_name])

        candidates_df = pd.DataFrame(candidates_dict)

        candidates_df["score"] = candidates_df["logprob"] + candidates_df["lang"] + candidates_df["accuracy"] + candidates_dict["parsability"]

        candidates_df = candidates_df.sort_values(by="score", ascending=False)

        if len(candidates_df[candidates_df["score"] > 0]) == 0:
            return "None"
        
        return candidates_df.iloc[0]['candidate']
    except:
        return "None"


def construct_dataframe(
    data_path,
    output_path=None,
    lang="en",
    use_en_solution=False,
    use_translated_solution=False,
    en_path=None,
    task="deepscaler_train",
    ):
    
    train_dataset = TASK_LIST[task]()
    x, y = zip(*[train_dataset.dataset.__getitem__(idx) for idx in range(train_dataset.dataset.__len__())])

    df = pd.DataFrame()
    # prompt_fn = TASK_LIST[f"{lang}_reason"].user_message
    prompt_fn = lambda x: x
    system_message = [{"role": "system", "content": TASK_LIST[f"{lang}_system"].system_message}]

    if lang != "en":
        query_path = os.path.join(data_path, f"raw_traces/{task}+{lang}+translated+queries/")
        query_df = pd.read_json(os.path.join(query_path, "output.jsonl"), lines=True)

        query_df['input_candidates'] = query_df.apply(lambda row: from_key(row, "answer"), axis=1)
        query_df['problem'] = query_df.apply(
            lambda row: select_best_candidate(
                row, 
                col_name="input_candidates",
                # use_logprob=True,
                use_logprob=False,
                use_accuracy=False,
                use_lang=True,
            ), 
            axis=1
        )
        query_df["answer"] = query_df["ground_truth"]
    else:
        query_df = pd.DataFrame({
            "problem": x,
            "answer": y,
        })

    if use_translated_solution:
        lang_solution_path = os.path.join(data_path, f"raw_traces/{task}+{lang}+translated+solutions/")
        lang_solution_df = pd.read_json(os.path.join(lang_solution_path, "output.jsonl"), lines=True)
        lang_solution_df['solution_candidates'] = lang_solution_df.apply(lambda row: from_key(row, "answer"), axis=1)
        lang_solution_df['solution'] = lang_solution_df.apply(
            lambda row: select_best_candidate(
                row, 
                col_name="solution_candidates",
                # use_logprob=True,
                use_logprob=False,
                use_accuracy=False,
                use_lang=True,
                # use_parsability=True,
            ), 
            axis=1
        )
        # query_df['translated_solution'] = lang_solution_df['solution']
        query_df['translated_solution'] = lang_solution_df.apply(lambda x: fix_answer(x, lang=lang, answer_key="ground_truth"), axis=1)
    else:
        query_df['translated_solution'] = "None"

    if use_en_solution:

        if en_path:
            en_solution_path = en_path
        else:
            en_solution_path = os.path.join(data_path, f"raw_traces/{task}+{lang}+generated+traces/")
        en_solution_df = pd.read_json(os.path.join(en_solution_path, "output.jsonl"), lines=True)

        en_solution_df['solution_candidates'] = en_solution_df.apply(lambda row: from_key(row, "answer"), axis=1)
        en_solution_df['solution'] = en_solution_df.apply(
            lambda row: select_best_candidate(
                row, 
                col_name="solution_candidates",
                use_logprob=True,
                use_accuracy=True,
                use_lang=False,
            ), 
            axis=1
        )
        query_df['solution'] = en_solution_df['solution']
    else:
        dataset_df = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train").to_pandas()
        # query_df['solution'] = None
        query_df['solution'] = dataset_df.apply(lambda x: fix_answer(x), axis=1)
        query_df['query'] = dataset_df['problem']

    df['solution'] = query_df['solution']
    df['translated_solution'] = query_df['translated_solution']
    df['reward_model'] = query_df.apply(
        lambda row: {
            "ground_truth": str(row["answer"])
        },
        axis=1
    )

    df['query'] = query_df['query'] 
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
        lambda row: row["input"] + [{"role": "assistant", "content": row["translated_solution"]}],
        axis=1
    )

    train_df = df[(df["solution"] != "") & (df["solution"] != "None")]
    # test_df = df.iloc[-1000:]
    test_df = df.iloc[-100:]

    if lang != "en":
        task_list = ["math500", "mgsm", "global_mmlu", "belebele", "mt_math100"]
    else:
        task_list = ["math500"]

    for task_name in task_list:
        if task_name == "math500":
            full_task_name = task_name
            task_prompt_fn = TASK_LIST["en_reason"].user_message
        else:
            full_task_name = f'{task_name}_{lang}'
            task_prompt_fn = prompt_fn

        eval_df = pd.DataFrame()

        try:
            eval_dataset = TASK_LIST[full_task_name]()
            x, y = zip(*[eval_dataset.dataset.__getitem__(idx) for idx in range(eval_dataset.dataset.__len__())])
        except:
            print(f"Task {full_task_name} not found.")
            continue

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
        eval_df = eval_df.iloc[:100]
        test_df = pd.concat([test_df, eval_df], ignore_index=True)

    if output_path is None:
        output_path = os.path.join(data_path, f"prep_traces/{task}+{lang}/")
    os.makedirs(output_path, exist_ok=True)
    train_df.to_parquet(os.path.join(output_path, "train.parquet"))
    # valid_df.to_parquet(os.path.join(output_path, "valid.parquet"))
    test_df.to_parquet(os.path.join(output_path, "test.parquet"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--lang', type=str, default="en")
    parser.add_argument('--use_en_solution', action='store_true', default=False, help="Use english solutions")
    parser.add_argument('--use_translated_solution', action='store_true', default=False, help="Use translated solutions")
    parser.add_argument('--en_path', type=str, default=None)
    args = parser.parse_args()
    construct_dataframe(
        data_path=args.data_path,
        output_path=args.output_path,
        lang=args.lang,
        use_en_solution=args.use_en_solution,
        en_path=args.en_path,
        use_translated_solution=args.use_translated_solution,
    )
