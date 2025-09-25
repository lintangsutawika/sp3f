# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from copy import deepcopy
from pprint import pprint

import random
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.reward import compute_reward
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import postprocess_data
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss

from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)

from yeval.response.math_responses import get_boxed_answer

import asyncio
from openai import AsyncOpenAI, AsyncAzureOpenAI

LLM_URL = os.environ.get('CMU_URL')
LLM_KEY = os.environ.get('CMU_KEY')
# client = AsyncOpenAI(base_url=LLM_URL, api_key=LLM_KEY)
client = AsyncAzureOpenAI(
    azure_endpoint=LLM_URL,
    api_key=LLM_KEY,
    api_version="2024-12-01-preview"
)

LANGUAGE_CODE = {
    "en": "English",
    "zh": "Chinese",
    "es": "Spanish",
    "sw": "Swahili",
    "id": "Indonesian",
    "ja": "Japanese",
}

pairwise_system_message = lambda x: f"""For the following Query, you will be given a solution and two thinking responses.
Query Start
{x["query"]}
Query End

Solution START
{x["original"]}
Solution END

Response A START
{x["A"]}
Response A END

Response B START
{x["B"]}
Response B END

Identify major and minor errors in response A and B and use the solution as a reference. At the end, choose which is better. \
Think step by step and answer with either \\boxed{{A}} or \\boxed{{B}}.\
"""
# Think step by step and answer with either \\boxed{{A}} or \\boxed{{B}}. If both responses are not satisfying, you can answer with \\boxed{{Neither}}.\

# pairwise_system_message = lambda x: f"""You will be given a solution and two thinking responses.
# Solution START
# {x["original"]}
# Solution END

# Thinking A START
# {x["A"]}
# Thinking A END

# Thinking B START
# {x["B"]}
# Thinking B END

# Between thinking responses \\boxed{{A}} or \\boxed{{B}} which is aligned the most with the Solution?""" + "Think step by step and answer with either \\boxed{{A}} or \\boxed{{B}}./nothink"

language_system_message = """\
You are a helpful assistant. You will be given a response and classify what language it is in. \
If the response is in English, output "en". \
If the response is in Chinese, output "zh". \
"""

reason_system_message="Reason step by step and put your final answer within \\boxed{}."


def get_answer(x):
    return get_boxed_answer("\\boxed{" + x)

class CustomRayPPOTrainer(RayPPOTrainer):

    def re_init_reward_model(self, model_path):
        self.rm_wg.config.model = model_path
        self.rm_wg.init_model()

    # @property
    # def get_rm_model_path(self):
    #     return self.rm_wg.config.model

    def _save_checkpoint(self):
        """Save checkpoint by calling the parent class method"""
        super()._save_checkpoint()

        if self.config.trainer.get("use_gcs", False):
            import fsspec
            import shutil

            fs = fsspec.filesystem("gcs",
                project=self.config.trainer.gcs_project,
                token=self.config.trainer.gcs_token
            )

            for file_name in os.listdir(self.config.trainer.validation_data_dir):
                fs.put(
                    os.path.join(self.config.trainer.validation_data_dir, file_name),
                    os.path.join(self.config.trainer.gcs_path, ""),
                    recursive=True
                )

                local_global_step_folder = os.path.join(
                    self.config.trainer.validation_data_dir, file_name
                )
                
                shutil.rmtree(local_global_step_folder, ignore_errors=True)


            for file_name in [f"global_step_{self.global_steps}", "latest_checkpointed_iteration.txt"]:
                fs.put(
                    os.path.join(self.config.trainer.default_local_dir, file_name),
                    os.path.join(self.config.trainer.gcs_path, ""),
                    recursive=True
                )

                local_global_step_folder = os.path.join(
                    self.config.trainer.default_local_dir, file_name
                )
                
                shutil.rmtree(local_global_step_folder, ignore_errors=True)


class RayGRPOTrainer(CustomRayPPOTrainer):

    def __post_init__(self):
        if not os.path.exists(self.config.trainer.default_local_dir):
            os.makedirs(self.config.trainer.default_local_dir, exist_ok=True)

    def _switch_chat_template(
        self,
        data: DataProto,
        n_rollouts=None,
        n_compare=None,
        system_message=None,
        check_for_boxed_content=False,
        tokenize=True,
    ):
        
        tgt_lang_code = self.config.trainer.lang_code
        tgt_lang_name = LANGUAGE_CODE[tgt_lang_code]

        src_max_length = data.batch["attention_mask"].shape[-1]
        # the maximum length is actually determined by the reward model itself
        max_length = self.config.get("max_length", src_max_length)
        if max_length is None:
            max_length = src_max_length
        # print(f"max_length: {max_length}")

        # system_message += "\nThink step by step before answering and output your answer in \\boxed{}."
        range_bs = list(range(data.batch.batch_size[0]))

        if (n_rollouts is not None) and (n_compare is not None):
            pairwise_chunks = [range_bs[i:i+n_rollouts] for i in range(0, len(range_bs), n_rollouts)]

            pairwise_idx = []
            for chunk in pairwise_chunks:
                for i in chunk:
                    _chunk = [j for j in chunk if j != i]
                    n_compare = min(n_compare, len(_chunk))
                    pairwise_idx.extend([(i,j) for j in random.sample(_chunk, n_compare)])

            chat_list = []
            for idx, (i, j) in enumerate(pairwise_idx):
                # extract raw prompt
                en_query = data.non_tensor_batch["query"][i]
                en_response = data.non_tensor_batch["solution"][i]

                response_dict = {}
                skip_judgement = 0
                for idx_resp, x in zip(["A", "B"],[i, j]):
                    # extract response
                    response_ids = data.batch["responses"][x]
                    response_length = response_ids.shape[-1]
                    valid_response_length = data.batch["attention_mask"][x][-response_length:].sum()
                    valid_response_ids = response_ids[:valid_response_length]

                    # decode
                    response = self.tokenizer.decode(valid_response_ids)
                    # remove bos and eos
                    response = response.replace(self.tokenizer.eos_token, "")

                    # if check_for_boxed_content:
                    #     # box_content = get_boxed_answer(response)
                    #     box_content = get_answer(response)
                    #     if box_content != "None":
                    #         response = response.replace(f"\\boxed{{{box_content}}}", box_content)

                    if tokenize:
                        if "</think>" in response:
                            thinking_part = response.split("</think>")[0] + "</think>"
                        else:
                            # thinking_part = "<think>" + "</think>"
                            thinking_part = "<think>Empty Response</think>"
                            skip_judgement += 1
                        response_dict[idx_resp] = thinking_part
                    else:
                        response_dict[idx_resp] = response

                x = {
                    "language": tgt_lang_name,
                    "query": en_query,
                    "original": en_response,
                    "A": response_dict["A"],
                    "B": response_dict["B"],
                }

                if tokenize:
                    content = system_message(x) + "/no_think"
                else:
                    content = system_message(x)

                if skip_judgement == 2:
                    # Skip
                    chat_list.append([
                        {"role": "user", "content": "<|im_end|>/no_think say 'hi'."}
                    ])
                else:
                    chat_list.append([
                        # {"role": "system", "content": system_message},
                        {"role": "user", "content": content},
                        # system_message(x)
                    ])

        else:

            chat_list = []
            for idx in range_bs:
                # extract raw prompt
                query = data.batch["query"][idx]
                response_ids = data.batch["responses"][idx]
                response_length = response_ids.shape[-1]
                valid_response_length = data.batch["attention_mask"][idx][-response_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                # decode
                response = self.tokenizer.decode(valid_response_ids)
                # remove bos and eos
                response = response.replace(self.tokenizer.eos_token, "")
                thinking_part = response.split("</think>")[0] + "</think>"

                chat_list.append([
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": thinking_part},
                ])

        if self.config.trainer.get("debug", False):
            _df = pd.DataFrame(
                data={
                    "chat_list": chat_list,
                }
            )
            _df.to_json(
                os.path.join(self.config.trainer.default_local_dir, f"judge_inputs_{self.global_steps}.json"),
                orient="records",
                lines=True,
            )

        if tokenize == False:
            return chat_list

        rm_input_ids = []
        rm_attention_mask = []
        for chat in chat_list:

            # if (n_rollouts is not None) and (n_compare is not None):
            #     prompt_with_chat_template = chat
            # else:
            #     prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=False, tokenize=False)
            prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=False, tokenize=False)
            model_inputs = self.tokenizer(prompt_with_chat_template, return_tensors="pt", add_special_tokens=False)
            input_ids, attention_mask = postprocess_data(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_length=max_length-1,
                pad_token_id=self.tokenizer.pad_token_id,
                # left_pad=False,  # right padding
                left_pad=True,  # right padding
                # truncation=self.config.get("truncation", "right"),
                truncation=self.config.get("truncation", "left"),
            )  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {"input_ids": rm_input_ids, "attention_mask": rm_attention_mask, "position_ids": rm_position_ids}

        return DataProto.from_dict(rm_inputs)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking
        import json

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        if not os.path.exists(self.config.trainer.default_local_dir):
            os.makedirs(self.config.trainer.default_local_dir, exist_ok=True)

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                do_profile = self.global_steps in self.config.trainer.profile_steps if self.config.trainer.profile_steps is not None else False
                if do_profile:
                    self.actor_rollout_wg.start_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.start_profile()
                    if self.use_critic:
                        self.critic_wg.start_profile()
                    if self.use_rm:
                        self.rm_wg.start_profile()

                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # Using LM Judge to compute pairwise comparison reward
                    with marked_timer("reward", timing_raw, color="yellow"):

                        def _expand_to_token_level(data: DataProto, scores: torch.Tensor):
                            batch_size = data.batch.batch_size[0]
                            # expand as token_level_reward
                            attention_mask = data.batch["attention_mask"]
                            position_ids = data.batch["position_ids"]
                            response_length = data.batch["responses"].shape[-1]
                            if position_ids.dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
                                position_ids = position_ids[:, 0, :]
                            eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
                            token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
                            token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

                            # select the response part
                            token_level_scores = token_level_scores[:, -response_length:]

                            return token_level_scores
                        
                        if self.config.trainer.get("use_privileged", False):
                            judge_batch = self._switch_chat_template(
                                batch,
                                n_rollouts=self.config.actor_rollout_ref.rollout.n,
                                n_compare=self.config.actor_rollout_ref.rollout.compare,
                                system_message=pairwise_system_message,
                                tokenize=not self.config.trainer.get("use_api_judge", False)
                                )
                            n_rollouts = self.config.actor_rollout_ref.rollout.n
                            n_compare = self.config.actor_rollout_ref.rollout.compare

                            # if self.global_steps == 1:
                            #     print("batch size", batch.batch.batch_size[0])
                            #     print("judge_batch size", judge_batch.batch.batch_size[0])
                            #     print("n_rollouts size", n_rollouts)
                            #     print("n_compare size", n_compare)

                            if self.config.trainer.get("use_api_judge", False):
                                judge_sampling_params = {
                                    # "stop": ["}"],
                                    # "extra_body": {
                                    #     "include_stop_str_in_output": True,
                                    #     "guided_choice": ["A}", "B}"],
                                    # },
                                }

                                async def get_judgement(idx, messages, **sampling_params):
                                    try:
                                        response = await client.chat.completions.create(
                                            # model="Qwen/Qwen3-8B",
                                            # model="azure/gpt-5-nano",
                                            # model="azure/gpt-5-mini",
                                            model=self.config.trainer.get("judge_model", "azure/o4-mini"),
                                            messages=messages,
                                            **sampling_params,
                                        )
                                        # print([resp.text for resp in response.choices][:10])
                                        # return [resp.text for resp in response.choices]
                                        return [idx, response.choices[0].message.content]
                                    except Exception as e:
                                        # print(f"Error in judgement: {e}")
                                        return [idx, ""]

                                # async def run_api(all_messages, **sampling_params):
                                #     tasks = [get_judgement(idx, messages, **sampling_params) for idx, messages in enumerate(all_messages)]
                                #     results = await asyncio.gather(*tasks)
                                #     return results

                                def chunk(lst, n):
                                    """Yield successive n-sized chunks from lst."""
                                    for i in range(0, len(lst), n):
                                        yield lst[i:i + n]

                                async def run_api(prompts, **sampling_params):
                                    all_results = []
                                    for chunk_idx, chunk_batch in enumerate(chunk(prompts, 128)):
                                        print("chunk_idx")
                                        response = [get_judgement(idx, messages, **sampling_params) for idx, messages in enumerate(chunk_batch)]
                                        results = await asyncio.gather(*response)
                                        await asyncio.sleep(10)  # to avoid rate limit
                                        results.sort(key=lambda x: x[0])
                                        all_results.extend([resp[1] for resp in results])
                                        # all_results.extend(results)
                                    return all_results

                                print("Using API judge")
                                judge_responses = asyncio.run(run_api(judge_batch, **judge_sampling_params))
                            else:

                                judge_batch.meta_info["judge"] = True
                                judge_sampling_params = {
                                    "detokenize": True,
                                    "temperature": 1.0,
                                    "do_sample": True,
                                    # "stop": ["}"],
                                    # "include_stop_str_in_output": True,
                                    # "guided_choice": ["A}", "B}"],
                                }
                                judge_batch.meta_info["sampling_params"] = judge_sampling_params
                                # judge_batch.meta_info["sampling_params"] = {}
                                judge_output = self.actor_rollout_wg.generate_sequences(judge_batch)
                                judge_responses = self.tokenizer.batch_decode(judge_output.batch["responses"], skip_special_tokens=True)

                            batch_size = list(range(batch.batch.batch_size[0]))
                            print("len(batch_size)", len(batch_size))
                            pairwise_chunks = [batch_size[i:i+n_rollouts] for i in range(0, len(batch_size), n_rollouts)]

                            pairwise_idx = []
                            for chunk in pairwise_chunks:
                                for i in chunk:
                                    _chunk = [j for j in chunk if j != i]
                                    n_compare = min(n_compare, len(_chunk))
                                    pairwise_idx.extend([(i,j) for j in random.sample(_chunk, n_compare)])

                            response_idx = {}
                            _idx = 0
                            for (i, j), response in zip(pairwise_idx, judge_responses):
                                try:
                                    winning_response = get_boxed_answer(response)
                                    # winning_response = get_answer(response)
                                except Exception as e:
                                    # print(f"Error in getting boxed answer: {e}")
                                    winning_response = "None"
                                if winning_response == "A":
                                    score = 1.0
                                elif winning_response == "B":
                                    score = 0.0
                                else:
                                    score = 0.0
                                if i in response_idx:
                                    response_idx[i].append(score)
                                else:
                                    response_idx[i] = [score]
                            
                            if self.config.trainer.get("debug", False):

                                response_idx_df = pd.DataFrame([
                                    {"response_idx": k, "scores": v} 
                                    for k, v in response_idx.items()
                                ])

                                response_idx_df.to_json(
                                    os.path.join(self.config.trainer.default_local_dir, f"response_idx_{self.global_steps}.json"),
                                    orient="records",
                                    lines=True,
                                )

                                response_df = pd.DataFrame(
                                    data={
                                        "responses": judge_responses,
                                    }
                                )

                                response_df.to_json(
                                    os.path.join(self.config.trainer.default_local_dir, f"judge_responses_{self.global_steps}.json"),
                                    orient="records",
                                    lines=True,
                                )

                            response_scores = torch.tensor([response_idx[i] for i in range(len(response_idx))], dtype=torch.float32).mean(dim=-1)
                            # response_scores = torch.tensor([response_idx[i] for i in range(len(response_idx))], dtype=torch.float32).sum(dim=-1)
                            # np.save(os.path.join(self.config.trainer.default_local_dir, f"judge_scores_{self.global_steps}.npy"), torch.tensor([response_idx[i] for i in range(len(response_idx))], dtype=torch.float32).cpu().numpy())
                            token_level_scores = _expand_to_token_level(batch, response_scores)
                            reward_tensor = token_level_scores.to("cpu")
                        else:

                            tgt_lang_code = self.config.trainer.lang_code
                            tgt_lang_name = LANGUAGE_CODE[tgt_lang_code]
                            judge_batch = self._switch_chat_template(
                                batch,
                                n_rollouts=None,
                                n_compare=None,
                                system_message=language_system_message+f"If the response is in {tgt_lang_name}, output \"{tgt_lang_code}\".",
                                check_for_boxed_content=True,
                                )

                            judge_batch.meta_info["validate"] = True
                            judge_output = self.actor_rollout_wg.generate_sequences(judge_batch)
                            judge_responses = self.tokenizer.batch_decode(judge_output.batch["responses"], skip_special_tokens=True)

                            response_list = []
                            for response in judge_responses:
                                lang_response = get_boxed_answer(response)
                                if lang_response is None:
                                    lang_response = response.strip()
                                score = 1.0 if lang_response == tgt_lang_code else 0
                                response_list.append(score)

                            response_scores = torch.tensor(response_list, dtype=torch.float32)
                            token_level_scores = _expand_to_token_level(batch, response_scores)
                            reward_tensor = token_level_scores.to("cpu")

                        reward_tensor_from_fn, _ = compute_reward(batch, self.reward_fn)
                        if self.config.trainer.get("use_reward_fn", False):
                            # Save the combined reward tensor for debugging/analysis
                            if self.config.trainer.get("debug", False):
                                torch.save(
                                    reward_tensor_from_fn,
                                    os.path.join(self.config.trainer.default_local_dir, f"reward_tensor_from_fn_{self.global_steps}.pt")
                                )

                                torch.save(
                                    reward_tensor,
                                    os.path.join(self.config.trainer.default_local_dir, f"reward_tensor{self.global_steps}.pt")
                                )

                            # reward_tensor += reward_tensor_from_fn
                            reward_tensor = 0.5*reward_tensor + 0.5*reward_tensor_from_fn
                        else:
                            reward_tensor = reward_tensor_from_fn

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        # reward_extra_infos_dict: dict[str, list]
                        # if self.config.reward_model.launch_reward_fn_async:
                        #     reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        reward_extra_infos_dict = {}
                        batch.batch["token_level_scores"] = reward_tensor

                        # if reward_extra_infos_dict:
                        #     batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            # multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    esi_close_to_expiration = should_save_ckpt_esi(max_steps_duration=self.max_steps_duration, redundant_time=self.config.trainer.esi_redundant_time)
                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if do_profile:
                    self.actor_rollout_wg.stop_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.stop_profile()
                    if self.use_critic:
                        self.critic_wg.stop_profile()
                    if self.use_rm:
                        self.rm_wg.stop_profile()

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
