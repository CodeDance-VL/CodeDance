# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import re

from mathruler.grader import grade_answer


def format_reward(predict_str: str) -> float:
    """
    Checks if the prediction has the correct format for a multi-turn dialogue.

    A correct format is defined by three conditions:
    1. A final answer must be present within <answer> tags.
    2. The number of <tool_call> blocks must equal the number of <tool_response> blocks.
    3. The number of opening <think> tags must equal the number of closing </think> tags.
    """
    # 1. Check for the presence of a well-formed answer tag.
    if not re.search(r"<answer>.*?</answer>", predict_str, re.DOTALL):
        return 0.0

    # 2. The number of <tool_call> blocks must equal the number of <tool_response> blocks.
    num_tool_calls = len(re.findall(r"<tool_call>.*?</tool_call>", predict_str, re.DOTALL))
    num_tool_responses = len(re.findall(r"<tool_response>.*?</tool_response>", predict_str, re.DOTALL))
    if num_tool_calls != num_tool_responses:
        return 0.0

    # 3. The number of opening <think> tags must equal the number of closing </think> tags.
    num_open_thinks = predict_str.count("<think>")
    num_close_thinks = predict_str.count("</think>")
    if num_open_thinks != num_close_thinks:
        return 0.0

    return 1.0


def tool_reward(predict_str: str) -> float:
    """
    Calculates a reward based on the success of tool calls.
    Penalizes responses that indicate an error or no output.
    """
    tool_responses = re.findall(r"<tool_response>(.*?)</tool_response>", predict_str, re.DOTALL)
    num_tool_calls = len(tool_responses)

    bad_call_count = 0
    error_strings = [
        "Code executed successfully. No explicit output or figures produced.",
        "Error",
    ]
    for response in tool_responses:
        if any(error_str in response for error_str in error_strings):
            bad_call_count += 1
    if num_tool_calls == 0:
        return 0.0
    # Using a smoothed ratio to calculate reward.
    # This avoids division by zero and gives a reward of 1 if no tools are called,
    # though format_reward gate would likely make score 0 in that case.
    reward = (num_tool_calls - bad_call_count) / (num_tool_calls)
    return reward


def multiturn_reward(predict_str: str) -> float:
    """
    Calculates a reward based on the number of user turns.
    """
    user_turns = predict_str.count("\nuser\n")
    return user_turns / 6.0


def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    if use_boxed:
        answer = extract_boxed_content(predict_str)
    else:
        answer = predict_str
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(predict_str: str, ground_truth: str, use_boxed: bool = True, format_score: float = 0.1) -> float:
    return (1.0 - format_score) * acc_reward(predict_str, ground_truth, use_boxed) + format_score * format_reward(
        predict_str
    )
