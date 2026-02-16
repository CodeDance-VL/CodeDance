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
import numpy as np

from math_verify import parse, verify


def normalize_answer(answer: str) -> str:
    if answer is None:
        return None
    if "dfrac" in answer:
        answer = answer.replace("dfrac", "frac") 
    if "text" in answer:
        answer = answer.replace("\\text", "")
    if "\\varnothing" in answer:
        answer = answer.replace("\\varnothing", "\\emptyset")
    if "minutes" in answer:
        answer = answer.replace("minutes", "")
    if "cm" in answer:
        answer = answer.replace("cm", "")
    return answer


def do_verify(nsol, b):
    res = 0.0
    try:
        a = parse(nsol)
        if len(b) > 1 and (b[1] in "ABCDEFGHIJK"):
            res = float(nsol[len("\\boxed{") :].startswith(b[1]))
        else:
            if len(a) == 0:
                res = 0.0
            else:
                res = float(verify(a, b))
    except Exception:
        res = -1.0
    return res


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
        "timeout",
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
    if isinstance(ground_truth, (list, tuple)):
        gt_list = [str(gt).lower() for gt in ground_truth]
        has_percent = None
        for gt in gt_list:
            if "%" in gt:
                has_percent = gt.replace("%", "")
                break
        if has_percent is not None and has_percent not in gt_list:
            gt_list.insert(0, has_percent)
    else:
        gt_list = [str(ground_truth).lower()]
    sol_lower = predict_str.lower()
    res = 0
    if len(gt_list) == 0:
        print(ground_truth)
    for gt in gt_list:
        match = re.search(r"\\boxed\{\s*\[\s*'(.*)'\s*\]\s*\}", gt)
        if match:
            gt = match.group(1)
            
        normalized_gt = normalize_answer(gt)
        b = None
        try:
            if "\\boxed" in normalized_gt:
                b = parse(normalized_gt)
            else:
                b = parse(f"\\boxed{{{normalized_gt}}}")
        except Exception:
            continue
        
        if not b:
            continue
        res = 0.0
        for indicator in ["\\boxed", "<answer>", "answer:"]:
            if indicator in sol_lower:
                nsol = ""
                if indicator == "<answer>":
                    found = re.search(r"<answer>(.*?)</answer>", sol_lower, re.DOTALL)
                    if found:
                        nsol = f"\\boxed{{{found.group(1).strip()}}}"
                    else:
                        continue
                elif indicator == "answer:":
                    tmp = sol_lower.split(indicator)[-1].strip()
                    nsol = f"\\boxed{{{tmp}}}"
                else:  # \\boxed
                    boxed_index = sol_lower.rfind(indicator)
                    nsol = sol_lower[boxed_index:].strip()
                res = do_verify(normalize_answer(nsol), b)
                if res > 0.5:
                    break

        if res < 0.5:
            res = do_verify(normalize_answer(sol_lower), b)

    return res


def compute_score(solution_str: str, ground_truth: str, **kwargs) -> dict:
    acc = acc_reward(solution_str, ground_truth)
    format_r = format_reward(solution_str)
    tool_r = tool_reward(solution_str)

    # The final score is a weighted average of accuracy and tool reward,
    # gated by the format reward.
    score = format_r * 0.2 + acc + tool_r * (1.2) * acc

    return {"score": score, "acc": acc, "format": format_r, "tool": tool_r, "ground_truth": ground_truth}


if __name__ == "__main__":
    predict_str = """
<think>From the image, the distance between Lord's cricket ground and Melbourne Cricket Ground is given as 10,513.61 miles.</think>
<answer>10,513.61 miles</answer>
    """
    print(compute_score(predict_str, "\\boxed{['10,513.61 miles']}"))

