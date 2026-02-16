# Copyright 2025 Individual Contributor: Mert Unsal
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

from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn


@register("batch")
class BatchRewardManager(AbstractRewardManager):
    """
    A batch reward manager that computes rewards for a batch of data.

    Args:
        tokenizer (Tokenizer): The tokenizer to use for decoding the responses.
        num_examine (int): The number of responses to examine.
        compute_score (callable): The function to compute the rewards.
        reward_fn_key (str): The key to use for the reward function.
        reward_kwargs (dict): The keyword arguments to pass to the reward function.
    """

    def __init__(
        self, tokenizer, num_examine, compute_score: RawRewardFn, reward_fn_key="data_source", **reward_kwargs
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs

    def verify(self, data: DataProto) -> Union[List[float], List[Dict[str, Any]]]:
        """
        Computes the reward scores for the given batch of data.

        Args:
            data (DataProto): The input data containing prompts and responses.

        Returns:
            Union[List[float], List[Dict[str, Any]]]: A list of scores, where each score can be a float or a dictionary.
        """
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str = []
        for i in range(len(data)):
            valid_len = valid_response_lengths[i]
            valid_response_ids = response_ids[i][:valid_len]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            responses_str.append(response_str)

        ground_truths = [item.non_tensor_batch["reward_model"].get("ground_truth", None) for item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        rollout_reward_scores = data.non_tensor_batch.get("reward_scores", [{} for _ in range(len(data))])
        extras = data.non_tensor_batch.get("extra_info", [{} for _ in range(len(data))])

        for i in range(len(data)):
            extras[i]["rollout_reward_scores"] = rollout_reward_scores[i]

        scores = self.compute_score(
            data_sources=data_sources,
            solution_strs=responses_str,
            ground_truths=ground_truths,
            extra_infos=extras,
            **self.reward_kwargs,
        )

        return scores

    def __call__(self, data: DataProto, return_dict: bool = False) -> Union[torch.Tensor, Dict[str, Any]]:
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        tool_reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key]

        scores = self.verify(data)
        rewards = []
        already_printed: Dict[str, Any] = {}
        tool_step_rewards = []
        all_tool_values = []

        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            score = scores[i]

            if isinstance(score, dict):
                reward = score["score"]
                tool_reward = score.get("tool_reward_component", 0.0)
                tool_step_reward = score.get("tool_step", [])  # Get tool_step_reward list
                for key, value in score.items():
                    reward_extra_info[key].append(value)
                if "tool_step" in reward_extra_info.keys():
                    reward_extra_info.pop("tool_step")

            else:
                reward = score
                tool_reward = 0.0
                tool_step_reward = []

            rewards.append(reward)
            reward_tensor[i, length - 1] = reward
            tool_reward_tensor[i, length - 1] = tool_reward

            tool_step_rewards.append(tool_step_reward)

            all_tool_values.extend(tool_step_reward)

        multi_step_values = []
        for tool_step_reward in tool_step_rewards:
            if len(tool_step_reward) > 1:
                multi_step_values.extend(tool_step_reward[:-1])

        # Compute mean/std only if needed, but always ensure last step is zeroed
        mean_val = 0.0
        std_val = 0.0
        if multi_step_values:
            mean_val = sum(multi_step_values) / len(multi_step_values)
            variance = sum((x - mean_val) ** 2 for x in multi_step_values) / len(multi_step_values)
            std_val = variance**0.5

        normalized_tool_step_rewards = []
        for tool_step_reward in tool_step_rewards:
            if len(tool_step_reward) > 1 and std_val > 1e-8:
                normalized_reward = [(x - mean_val) / std_val for x in tool_step_reward[:-1]]
                normalized_reward.append(0.0)
                normalized_tool_step_rewards.append(normalized_reward)
            elif len(tool_step_reward) > 0:
                normalized_tool_step_rewards.append([*tool_step_reward[:-1], 0.0])
            else:
                normalized_tool_step_rewards.append(tool_step_reward)
        tool_step_rewards = normalized_tool_step_rewards

        # Highly optimized vectorized conversion of tool_step_rewards to tensor format
        tool_step_rewards_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        if "response_mask" in data.batch and tool_step_rewards:
            response_mask = data.batch["response_mask"].float()
            batch_size, seq_len = response_mask.shape
            device = response_mask.device

            # Vectorized computation of segment boundaries for all batches at once
            # Add padding to handle edge cases
            mask_padded = torch.cat(
                [torch.zeros(batch_size, 1, device=device), response_mask, torch.zeros(batch_size, 1, device=device)],
                dim=1,
            )

            # Find all segment starts and ends using diff
            mask_diff = mask_padded[:, 1:] - mask_padded[:, :-1]
            segment_starts = (mask_diff == 1).float()[:, :-1]  # 0->1 transitions

            # Create segment indices for each position
            segment_ids = torch.cumsum(segment_starts, dim=1) * response_mask

            # Process each batch (this loop is unavoidable due to variable step counts)
            for batch_idx in range(batch_size):
                if batch_idx < len(tool_step_rewards) and len(tool_step_rewards[batch_idx]) > 0:
                    batch_segment_ids = segment_ids[batch_idx]
                    step_rewards = tool_step_rewards[batch_idx]

                    # Create reward mapping for this batch
                    for step_idx, reward_val in enumerate(step_rewards):
                        # Find positions belonging to this segment (step_idx + 1 because cumsum starts from 1)
                        segment_mask = batch_segment_ids == (step_idx + 1)
                        tool_step_rewards_tensor[batch_idx] = torch.where(
                            segment_mask,
                            torch.tensor(reward_val, dtype=torch.float32, device=device),
                            tool_step_rewards_tensor[batch_idx],
                        )

        for i in range(len(data)):
            data_source = data_sources[i]
            if already_printed.get(data_source, 0) < self.num_examine:
                current_length = valid_response_lengths[i].item()
                response_str = self.tokenizer.decode(
                    data.batch["responses"][i][:current_length], skip_special_tokens=True
                )
                prompt_str = self.tokenizer.decode(data.batch["prompts"][i], skip_special_tokens=True)
                ground_truth = data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", scores[i])
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "tool_reward_tensor": tool_reward_tensor,
                "reward_extra_info": reward_extra_info,
                "tool_step_rewards_tensor": tool_step_rewards_tensor,
            }
        else:
            return reward_tensor
