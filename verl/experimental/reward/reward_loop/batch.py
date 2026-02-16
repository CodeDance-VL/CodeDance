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

import inspect
from collections import defaultdict

import torch

from verl import DataProto
from verl.experimental.reward.reward_loop import register
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase
from verl.utils.reward_score import default_compute_score


@register("batch")
class BatchRewardLoopManager(RewardLoopManagerBase):
    def __init__(self, config, tokenizer, compute_score=None, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = data_item.non_tensor_batch.get("extra_info", {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields.items())
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        if rollout_reward_scores:
            extra_info["rollout_reward_scores"] = rollout_reward_scores
        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )
        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )
        reward_extra_info = {}
        if isinstance(result, dict):
            score = result.get("score", 0.0)
            for key, value in result.items():
                reward_extra_info[key] = value
            tool_reward_component = result.get("tool_reward_component", 0.0)
            tool_step = result.get("tool_step", [])
        else:
            score = float(result)
            reward_extra_info["acc"] = score
            tool_reward_component = 0.0
            tool_step = []
        return {
            "reward_score": score,
            "reward_extra_info": reward_extra_info,
            "tool_reward_component": tool_reward_component,
            "tool_step": tool_step,
        }

    def __call__(self, data: DataProto, return_dict: bool = False):
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

        async def process_batch():
            import asyncio
            tasks = [self.run_single(data[i : i + 1]) for i in range(len(data))]
            return await asyncio.gather(*tasks)

        results = self.loop.run_until_complete(process_batch())

        rewards = []
        tool_step_rewards = []

        for i, result in enumerate(results):
            length = valid_response_lengths[i].item()
            reward = result["reward_score"]
            tool_reward = result.get("tool_reward_component", 0.0)
            step_list = result.get("tool_step", [])
            rewards.append(reward)
            reward_tensor[i, length - 1] = reward
            tool_reward_tensor[i, length - 1] = tool_reward
            tool_step_rewards.append(step_list)
            for key, value in result.get("reward_extra_info", {}).items():
                reward_extra_info[key].append(value)
            if "tool_step" in reward_extra_info:
                reward_extra_info.pop("tool_step")

        multi_step_values = []
        for step_list in tool_step_rewards:
            if len(step_list) > 1:
                multi_step_values.extend(step_list[:-1])

        mean_val = 0.0
        std_val = 0.0
        if multi_step_values:
            mean_val = sum(multi_step_values) / len(multi_step_values)
            variance = sum((x - mean_val) ** 2 for x in multi_step_values) / len(multi_step_values)
            std_val = variance ** 0.5

        normalized_tool_step_rewards = []
        for step_list in tool_step_rewards:
            if len(step_list) > 1 and std_val > 1e-8:
                normalized = [(x - mean_val) / std_val for x in step_list[:-1]]
                normalized.append(0.0)
                normalized_tool_step_rewards.append(normalized)
            elif len(step_list) > 0:
                normalized_tool_step_rewards.append([*step_list[:-1], 0.0])
            else:
                normalized_tool_step_rewards.append(step_list)
        tool_step_rewards = normalized_tool_step_rewards

        tool_step_rewards_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        if "response_mask" in data.batch and tool_step_rewards:
            response_mask = data.batch["response_mask"].float()
            batch_size, _ = response_mask.shape
            device = response_mask.device
            mask_padded = torch.cat(
                [
                    torch.zeros(batch_size, 1, device=device),
                    response_mask,
                    torch.zeros(batch_size, 1, device=device),
                ],
                dim=1,
            )
            mask_diff = mask_padded[:, 1:] - mask_padded[:, :-1]
            segment_starts = (mask_diff == 1).float()[:, :-1]
            segment_ids = torch.cumsum(segment_starts, dim=1) * response_mask
            for batch_idx in range(batch_size):
                if batch_idx < len(tool_step_rewards) and len(tool_step_rewards[batch_idx]) > 0:
                    batch_segment_ids = segment_ids[batch_idx]
                    step_rewards = tool_step_rewards[batch_idx]
                    for step_idx, reward_val in enumerate(step_rewards):
                        segment_mask = batch_segment_ids == (step_idx + 1)
                        tool_step_rewards_tensor[batch_idx] = torch.where(
                            segment_mask,
                            torch.tensor(reward_val, dtype=torch.float32, device=device),
                            tool_step_rewards_tensor[batch_idx],
                        )

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
