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


def acc_reward(predict_str: str, ground_truth: str) -> float:
    match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
    answer = match.group(1) if match else None
    if isinstance(ground_truth, (list, tuple)):
        ground_truth = ground_truth[0]

    # Extract effective part from ground_truth if it's in "\\boxed{B}" format
    boxed_match = re.search(r"\\boxed{(.*?)}", ground_truth)
    if boxed_match:
        ground_truth = boxed_match.group(1)
        
    return 1.0 if grade_answer(answer, ground_truth) else 0.0



def compute_score(solution_str: str, ground_truth: str, **kwargs) -> dict:
    acc = acc_reward(solution_str, ground_truth)
    format_r = format_reward(solution_str)
    tool_r = tool_reward(solution_str)

    # The final score is a weighted average of accuracy and tool reward,
    # gated by the format reward.
    score = format_r * 0.2 + acc + tool_r * 0.5

    return {"score": score, "acc": acc, "format": format_r, "tool": tool_r, "ground_truth": ground_truth}


if __name__ == "__main__":
    predict_str = """
I notice that the figure appears to be a trapezoid with several parallel lines and segments. Given the lengths provided, it seems like we're dealing with a problem involving similar triangles or proportional segments due to the parallel lines. However, without more context or specific relationships defined, I'll assume that the segments are proportional based on the parallel lines, which is a common scenario in such problems.\n\nThe key observation is that the segments $LK$, $MP$, $PQ$, $KJ$, $RS$, and $LP$ seem to be related in a way that suggests proportionality. Specifically, the segments $LK$, $MP$, and $PQ$ are likely to be proportional to the segments $RS$ and $LP$ because of the parallel lines.\n\nLet's calculate the ratio of $LK$ to $MP$ and see if it helps us find $QR$. If the segments are proportional, then the ratio of $LK$ to $MP$ should be the same as the ratio of $QR$ to $RS$.<tool_call>
{"name": "calc_execute_python", "arguments": {"code": "ratio = LK / MP"}}
</tool_call>
user
<tool_response>
--- Execution Error (Error) ---
NameError: name 'LK' is not defined
</tool_response>
assistant
It seems there's still an issue with referencing the variable $RS$. Since $RS=6$ is given, let's directly substitute this value into our equation:\n\n\\[
QR = \\left(\\frac{4}{3}\\right) \\times 6
\\]\n\nNow, let's calculate the value of $QR$.<tool_call>
{"name": "calc_execute_python", "arguments": {"code": "RS = 6\\nQR = (4/3) * RS"}}
</tool_call>
<tool_response>
--- Execution Result (Success) ---
QR = 8.0
</tool_response>
assistant
The calculation shows that:\n\n\\[
QR = \\left(\\frac{4}{3}\\right) \\times 6 = 8
\\]\n\nTherefore, the final answer is:\n\n<answer>8</answer>
    """
    print(compute_score(predict_str, "8"))