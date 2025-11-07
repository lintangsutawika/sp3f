import os
import random

import pandas as pd
# Which is more preferable?
# Swap positions A-B, B-A
import asyncio
from openai import AsyncOpenAI, AsyncAzureOpenAI
import re

from yeval.response.math_responses import get_boxed_answer

# LLM_URL = os.environ.get('CMU_URL')
# LLM_KEY = os.environ.get('CMU_KEY')
# client = AsyncOpenAI(base_url=LLM_URL, api_key=LLM_KEY)
# client = AsyncAzureOpenAI(
client = AsyncOpenAI(
    base_url="https://cmu.litellm.ai",
    api_key="sk-20_0UJNpT91Yq9BdLRX5Fw",
)

data_path = "/datadrive/lsutawik/lbr/data/eval_scores/"

SYSTEM_MESSAGE = """You are an expert judge in evaluating the quality of responses to user queries. \
Your task is to determine which response (A or B) is preferable. \
The responses may be in various languages. \
Write your analysis and end it by answering with either \\boxed{A} or \\boxed{B}.
"""
USER_MESSAGE = lambda x: f"""<Query>
{x["query"]}
</Query>

<Response A>
{x["response_a"]}
</Response A>

<Response B>
{x["response_b"]}
</Response B>""" + """
First, decide which of the two responses is preferable. \
Finally, choose which is better by answering with either \\boxed{{A}} or \\boxed{{B}}. \
You MUST provide your reasoning before the answer."""

model_list = [
    "r_privileged_API_GPT_4o_Mini-r_acc-r_threshold-r_parsable+sft-Qwen2.5-7B-deepscaler_train+deepscaler_train+LANGUAGE+0",
    # "r_nonprivileged_API_GPT_4o_Mini-r_acc-r_threshold-r_parsable+sft-Qwen2.5-7B-deepscaler_train+deepscaler_train+LANGUAGE+0",
    # "r_acc-r_threshold-r_parsable+sft-Qwen2.5-7B-deepscaler_train+deepscaler_train+LANGUAGE+0",
    "Qwen-Qwen2.5-7B-Instruct",
]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="id", help="Language to process")
    parser.add_argument("--model_a", type=str, default="", help="Model A to compare")
    parser.add_argument("--model_b", type=str, default="", help="Model B to compare")
    args = parser.parse_args()
    language = args.language
    model_a = args.model_a
    model_b = args.model_b
    if model_a != "" and model_b != "":
        model_list = [model_a, model_b]
    
    judge_model="azure/gpt-4o-mini"
    # judge_model="neulab/gpt-5-nano"

    async def get_judgement(idx, messages, **sampling_params):
        try:
            response = await client.chat.completions.create(
            # response = client.chat.completions.create(
                # model="neulab/gpt-5-nano",
                # model="azure/gpt-5",
                model=judge_model,
                # model="azure/gpt-4.1-nano",
                messages=messages,
                **sampling_params,
            )
            # print(response)
            # print([resp.text for resp in response.choices][:10])
            # return [resp.text for resp in response.choices]
            return [idx, response.choices[0].message.content]
        except Exception as e:
            print(f"Error in judgement: {e}")
            return [idx, ""]
        
    def chunk(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    async def run_api(prompts, **sampling_params):
        all_results = []
        for chunk_idx, chunk_batch in enumerate(chunk(prompts, 1024)):
            # print("chunk_idx")
            response = [get_judgement(idx, messages, **sampling_params) for idx, messages in enumerate(chunk_batch)]
            results = await asyncio.gather(*response)
            await asyncio.sleep(10)  # to avoid rate limit
            results.sort(key=lambda x: x[0])
            all_results.extend([resp[1] for resp in results])
            # all_results.extend(results)
        return all_results

    # for language in ["id", "es", "ja", "bn", "sw"]:
    # for language in ["ja", "es", "bn", "sw"]:
    judge_dict = {
        "model": [],
        "query": [],
        "task": [],
        "language": [],
        "response": [],
        "idx": [],
    }

    # for language in ["id"]:
    # language = "id"
    for model in model_list:
        model = model.replace("LANGUAGE", language)
        for task in ["mgsm", "mt_math100", "belebele", "global_mmlu"]:
        # for task in ["belebele", "global_mmlu"]:
            run = model+"+"+f"{task}_LANGUAGE+LANGUAGE_system".replace("LANGUAGE", language)
            print("run:", run)
            print("Loading data from:", os.path.join(data_path, run))
            df = pd.read_json(os.path.join(data_path, run, "output.jsonl"), lines=True)
            for _, row in df.iterrows():
                judge_dict["model"].append(model)
                judge_dict["query"].append(row["step"][0]["full_input"][-1]['content'])
                judge_dict["task"].append(task)
                judge_dict["language"].append(language)
                judge_dict["response"].append(row["answer"])
                judge_dict["idx"].append(row["idx"])
            # break

    judge_df = pd.DataFrame(data=judge_dict)


    for task in judge_df["task"].unique():
        print("Processing task:", task)

        pivot_judge_df = judge_df[judge_df["task"] == task].pivot_table(
            index=["idx"],
            columns=["model"],
            values=["query", "response"],
            aggfunc="first"
        )

        models = judge_df.model.unique()
        choice = ["a", "b"]
        model_dict = {model: choice[key] for key, model in enumerate(models)}
        # print(model_dict)
        pivot_judge_df.columns = [ '_'.join([str(c) for c in c_list]) for c_list in pivot_judge_df.columns.values ]
        pivot_judge_df.reset_index(inplace=True)

        N = 8

        query_dict = {
            "new_query": [],
        }
        for _, row in pivot_judge_df.iterrows():

            query = row[f"query_{models[0]}"]
            response_a = random.sample(row[f"response_{models[0]}"], N)
            response_b = random.sample(row[f"response_{models[1]}"], N)

            for i in range(N):

                for c in range(2):
                    if c == 0:
                        _response_a = response_a[i]
                        _response_b = response_b[i]
                    else:
                        _response_a = response_b[i]
                        _response_b = response_a[i]

                    instance = {
                        "query": query,
                        "response_a": _response_a,
                        "response_b": _response_b,
                    }
                    query_dict["new_query"].append([
                        {"role": "system", "content": SYSTEM_MESSAGE},
                        {"role": "user", "content": USER_MESSAGE(instance)}
                        ])
            # break

        query_df = pd.DataFrame(data=query_dict)
        # judge_responses = await run_api(query_df["new_query"].tolist(), **{})
        judge_responses = asyncio.run(run_api(query_df["new_query"].tolist(), **{}))
        query_df["judgement"] = judge_responses
        def boxed_answer_extractor(text):
            try:
                return get_boxed_answer(text)
            except Exception as e:
                print("Error extracting boxed answer:", e)
                return ""
        query_df["answer"] = query_df["judgement"].apply(boxed_answer_extractor)

        for model in models:
            pivot_judge_df[model] = 0.0

        for idx in range(len(pivot_judge_df)):

            score_dict = {
                k: 0.0 for k in model_dict.keys()
            }

            idx_df = query_df.iloc[idx* N * 2: (idx+1) * N * 2]
            for ans in idx_df["answer"][:N].tolist():
                if ans == "A":
                    score_dict[models[0]] += 1
                elif ans == "B":
                    score_dict[models[1]] += 1

            for ans in idx_df["answer"][N:].tolist():
                if ans == "B":
                    score_dict[models[1]] += 1
                elif ans == "A":
                    score_dict[models[0]] += 1

            pivot_judge_df.loc[idx, models[0]] = score_dict[models[0]]/ (N * 2)
            pivot_judge_df.loc[idx, models[1]] = score_dict[models[1]]/ (N * 2)

        judge_model_name = judge_model.replace("/", "-")
        output_dir = "data/pairwise_quality/"
        file_name = f"compiled_judgement-_-{task}-_-{judge_model_name}-_-{language}-_-{models[0][:30]}-_-{models[1][:30]}.jsonl"
        pivot_judge_df.to_json(
            os.path.join(output_dir, file_name),
            orient="records", lines=True
        )

        file_name = f"raw_judgement-_-{task}-_-{judge_model_name}-_-{language}-_-{models[0][:30]}-_-{models[1][:30]}.jsonl"
        query_df.to_json(
            os.path.join(output_dir, file_name),
            orient="records", lines=True
        )
        # break

        # print(f"Task: {task}")
        # display(pivot_judge_df[models].mean())