import requests
import random
import re
import os
from openai import OpenAI
import time
from math_verify import parse, verify
from concurrent.futures import ThreadPoolExecutor,as_completed
import re

MAX_WORKERS = 64
openai_api_key="EMPTY"
openai_api_base_str = os.environ.get("LLM_AS_A_JUDGE_BASE", "http://[Your_IP_HERE]:18901/v1")
openai_api_base_list = [base.strip() for base in openai_api_base_str.split(",")]


client_list = []
model_name_list = []

for api_base in openai_api_base_list:
    client = OpenAI(
        api_key=openai_api_key,
        base_url=api_base,
    )
    client_list.append(client)
model_name_list = ['judge']


def get_chat_template():
    chat_template = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""
    return chat_template

def get_gpt4_score_ICE():
    example_1 = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1
""" # noqa

    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
""" # noqa

    example_3 = """
[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
""" # noqa

    example_4 = """
[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
""" # noqa

    example_5 = """
[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
Judgement: 1
""" # noqa

    example_6 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
""" # noqa

    example_7 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]


MATH_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level math problems. I am tasked with evaluating the correctness of a student's answer. 
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Mathematical or Notational Equivalence: Pay special attention to any LaTeX expressions in both answers. Confirm that the mathematical relationships, variables, and operations conveyed are equivalent.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Question**:
{query}

**Reference Answer**
{gold_ans}

## Student Final Answer
{pred_ans}"""

def get_prompt(predict_str, ground_truth, question):
    examples = get_gpt4_score_ICE()
    chat_template = get_chat_template()
    demo_prompt = chat_template + "\n\n".join(examples)
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'
    return full_prompt

def extract_answer(text: str):
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def rule_math_verify(ground_truth, model_answer):
    gold = parse(ground_truth)
    answer = parse(model_answer)
    return verify(gold, answer)


def tool_reward_step(predict_str: str, format: float=0) -> list:
    """
    Calculates a reward for each tool call based on its response.
    Returns the cumulative discounted score as a list with variable length.
    Uses discount factor similar to PPO reward calculation.
    """
    tool_responses = re.findall(r"<tool_response>(.*?)</tool_response>", predict_str, re.DOTALL)
    scores = []
    error_strings = [
        "Code executed successfully. No explicit output or figures produced.",
        "Error",
        "timeout",
    ]
    for response in tool_responses:
        if any(error_str in response for error_str in error_strings):
            scores.append(-0.5)
        else:
            scores.append(0.0)
    scores.append(0)
    
    # Calculate cumulative discounted rewards (similar to PPO)
    discount_factor = 0.2  # Discount factor for future rewards
    cumulative_scores = [1.0] * len(scores)
    
    # Calculate from back to front (reverse order)
    cumulative_scores[-1] = scores[-1]
    for i in range(len(scores) - 2, -1, -1):
        cumulative_scores[i] = scores[i] + discount_factor * cumulative_scores[i + 1]
    
    # Return the cumulative scores list without padding
    return cumulative_scores

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

def count_tool_calls(predict_str: str) -> int:
    tool_responses = re.findall(r"<tool_response>(.*?)</tool_response>", predict_str, re.DOTALL)
    return len(tool_responses)


def generative_verify(query, ground_truth, model_answer):
    client_idx = random.randint(0, len(client_list) - 1)
    client = client_list[client_idx]
    model_name = model_name_list[client_idx]

    full_prompt = MATH_VERIFY_PROMPT.format(
        query=query,
        gold_ans=ground_truth,
        pred_ans=model_answer,
    )

    response = ""
    for it in range(8):
        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": full_prompt},
                ],
                seed = random.randint(0, 1000000),
                temperature=0.0,
            )
            response = chat_response.choices[0].message.content.strip()
            break
        except Exception as e:
            print(f' [ERROR math] generative_verify error: {e}')
            continue
    
    judgement = response.split('## Equivalence Judgement')[-1].lower()
    if 'true' in judgement and 'false' not in judgement:
        return True
    elif 'false' in judgement and 'true' not in judgement:
        return False
    else:
        print(f' [ERROR math] verify bug output: ')

    

def compute_score_default(predict_str: str, ground_truth: str, extra_info: dict = None) -> dict:
    """
    Computes the score for a prediction based on accuracy, format, and tool usage.
    It randomly selects a client from the global client_list for evaluation.
    """
    is_format_error = False
    if predict_str.count("<think>") != predict_str.count("</think>"):
        is_format_error = True
    predict_no_think = predict_str.split('</think>')[-1].strip()
    if predict_str.count("<answer>") != 1 or predict_str.count("</answer>") != 1:
        is_format_error = True
    # if not (predict_str.rstrip().endswith("</answer>") or predict_str.rstrip().endswith("</answer>\n")):
    #     is_format_error = True
    answer_text = extract_answer(predict_str)
    if answer_text is None:
        answer_text = ""
        is_format_error = True
    
    question_text = extra_info.get('question', '') if extra_info else ''
    full_prompt = get_prompt(answer_text, ground_truth, question_text)
    
    client_idx = random.randint(0, len(client_list) - 1)
    client = client_list[client_idx]
    model_name = model_name_list[client_idx]
    
    response_content = None
    acc_reward = 0.0

    while True:
        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": full_prompt},
                ],
                seed=random.randint(0, 1000000),
                temperature=0.3,
            )
            response_content = chat_response.choices[0].message.content
            break
        except Exception as e:
            print(f"Fail! {e}")
            if "429" in str(e): 
                time.sleep(1)
                continue
            else: 
                if "400" in str(e):
                    print(e)
                break
    
    if response_content:
        response = response_content.strip()
        if 'Judgement:' in response:
            response = response.split('Judgement:')[-1].strip()

        if '1' in response:
            acc_reward = 1.0
        elif '0' in response:
            acc_reward = 0.0
        else:
            print(f' [WARNING] resp format error {response=}')
            acc_reward = 0.0
    else:
        acc_reward = 0.0

    if len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True

    R_accuracy = 1.0 
    format_penalty = -1.0 if is_format_error else 0.0

    tool_reward_step_value = tool_reward_step(predict_str, format_penalty)
    num_tool_calls = count_tool_calls(predict_str)
    tool_success_rate = tool_reward(predict_str) if num_tool_calls > 0 else float("nan")

    reward = acc_reward * R_accuracy

    tool_reward_component = 0.0
    if num_tool_calls > 0:
        if acc_reward > 0.5:
            tool_reward_component = 1.5 * tool_success_rate
        else:
            tool_reward_component = 0.5 * tool_success_rate

    score = reward + 0.2 * format_penalty

    return {
        "score": score, 
        "acc": acc_reward, 
        "tool_step": tool_reward_step_value,
        "tool_success_rate": tool_success_rate,
        "tool_reward_component":tool_reward_component,
        "format_r": format_penalty, 
        "ground_truth": ground_truth,
        "answer": answer_text
    }


def compute_score_math(predict_str: str, ground_truth: str, extra_info: dict | None = None) -> dict:
    extra_info = extra_info or {}

    is_format_error = predict_str.count("<think>") != predict_str.count("</think>")
    predict_no_think = predict_str.split("</think>")[-1].strip()
    if "tool_call" in predict_no_think:
        is_format_error = True
    
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_list = re.findall(answer_pattern, predict_no_think, flags=re.DOTALL)
    # if not (predict_str.rstrip().endswith("</answer>") or predict_str.rstrip().endswith("</answer>\n")):
    #     is_format_error = True
    model_answer, acc_reward = "", 0.0
    if not answer_list:
        is_format_error = True
    else:
        # if len(answer_list) > 1:
        #     is_format_error = True
        model_answer = answer_list[-1].strip()

        if rule_math_verify(ground_truth, model_answer):
            acc_reward = 1.0
        else:
            acc_reward = 1.0 if generative_verify(
                extra_info.get("question", ""), ground_truth, model_answer
            ) else 0.0

    format_penalty = -1.0 if is_format_error else 0.0

    R_accuracy = 1.2 
    tool_reward_step_value = tool_reward_step(predict_str, format_penalty)
    num_tool_calls = count_tool_calls(predict_str)
    tool_success_rate = tool_reward(predict_str) if num_tool_calls > 0 else float("nan")

    tool_reward_component = 0.0
    if num_tool_calls > 0:
        if acc_reward > 0.5:
            tool_reward_component = 2 * tool_success_rate
        else:
            tool_reward_component = 1 * tool_success_rate

    reward = acc_reward * R_accuracy

    score = reward + 0.4 * format_penalty

    return {
        "score": score,
        "acc": acc_reward,
        "tool_step": tool_reward_step_value,
        "tool_success_rate": tool_success_rate,
        "tool_reward_component":tool_reward_component,
        "format_r": format_penalty,
        "ground_truth": ground_truth,
        "answer": model_answer
    }


def compute_score_SA1B(predict_str: str, ground_truth: str, extra_info: dict = None) -> dict:
    """
    Computes the score for a prediction based on accuracy, format, and tool usage.
    It randomly selects a client from the global client_list for evaluation.
    """
    is_format_error = False
    if predict_str.count("<think>") != predict_str.count("</think>"):
        is_format_error = True
    if predict_str.count("<answer>") != 1 or predict_str.count("</answer>") != 1:
        is_format_error = True
    # if not (predict_str.rstrip().endswith("</answer>") or predict_str.rstrip().endswith("</answer>\n")):
    #     is_format_error = True
    answer_text = extract_answer(predict_str)
    if answer_text is None:
        answer_text = ""
        is_format_error = True
    
    question_text = extra_info.get('question', '') if extra_info else ''
    full_prompt = get_prompt(answer_text, ground_truth, question_text)
    
    client_idx = random.randint(0, len(client_list) - 1)
    client = client_list[client_idx]
    model_name = model_name_list[client_idx]
    
    response_content = None
    acc_reward = 0.0

    while True:
        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": full_prompt},
                ],
                seed=random.randint(0, 1000000),
                temperature=0.3,
            )
            response_content = chat_response.choices[0].message.content
            break
        except Exception as e:
            print(f"Fail! {e}")
            if "429" in str(e): 
                time.sleep(1)
                continue
            else: 
                if "400" in str(e):
                    print(e)
                break
    
    if response_content:
        response = response_content.strip()
        if 'Judgement:' in response:
            response = response.split('Judgement:')[-1].strip()

        if '1' in response:
            acc_reward = 1.0
        elif '0' in response:
            acc_reward = 0.0
        else:
            print(f' [WARNING] resp format error {response=}')
            acc_reward = 0.0
    else:
        acc_reward = 0.0

    if len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True

    R_accuracy = 1.0 
    format_penalty = -1.0 if is_format_error else 0.0

    tool_reward_step_value = tool_reward_step(predict_str, format_penalty)
    num_tool_calls = count_tool_calls(predict_str)
    tool_success_rate = tool_reward(predict_str) if num_tool_calls > 0 else float("nan")
    reward = acc_reward * R_accuracy

    score = reward + 0.2 * format_penalty
    tool_reward_component = 0.0
    if num_tool_calls > 0:
        if acc_reward > 0.5:
            tool_reward_component = 1.5 * tool_success_rate
        else:
            tool_reward_component = 0.5 * tool_success_rate
    return {
        "score": score, 
        "acc": acc_reward, 
        "tool_step": tool_reward_step_value,
        "tool_success_rate": tool_success_rate,
        "tool_reward_component":tool_reward_component,
        "format_r": format_penalty, 
        "ground_truth": ground_truth,
        "answer": answer_text
    }


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict = None, **kwargs) -> dict:
    if extra_info is None:
        extra_info = {}

    if data_source in ['vstar', 'vl_agent', 'chart']:
        return compute_score_default(solution_str, ground_truth, extra_info)
    elif data_source in ['thinklite_eureka', 'xince']:
        return compute_score_math(solution_str, ground_truth, extra_info)
    elif data_source or data_source.lower() in ['SA1B', 'sa1b']:
        return compute_score_SA1B(solution_str, ground_truth, extra_info)
    else:
        return compute_score_default(solution_str, ground_truth, extra_info)


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
    math_sources = {'thinklite_eureka', 'xince'}
    sa1b_source = {'SA1B'}
    num_tasks = len(data_sources)
    results = [None] * num_tasks
    
    future_to_index = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i, (data_source, solution_str, ground_truth, extra_info) in enumerate(zip(data_sources, solution_strs, ground_truths, extra_infos)):
            if data_source in math_sources:
                try:
                    results[i] = compute_score(data_source, solution_str, ground_truth, extra_info)
                except Exception as e:
                    print(f"[ERROR] Sequential math task at index {i} failed: {e}")
                    results[i] = {"score": 0, "error": str(e)} 
            else:
                future = executor.submit(compute_score, data_source, solution_str, ground_truth, extra_info)
                future_to_index[future] = i
            
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
            except Exception as e:
                print(f"[ERROR] Parallel task at index {index} failed: {e}")
                results[index] = {"score": 0, "error": str(e)} 

    return results

if __name__ == "__main__":
    if not client_list:
        print("Execution stopped because no clients were available.")
    else:
        predict_str = """
        <think>
        The user wants to know the sum of 1 and 1. This is a simple arithmetic operation. The result is 2.
        </think>
        <answer>
        2
        </answer>
        """
        ground_truth = "2"
        result = compute_score("thinklite_eureka",predict_str, ground_truth, extra_info={'question': '1+1=?',})
        print(result)

        predict_str_format_error = """
<think>The woman's shirt appears to be a shade of purple based on the visible color in the image.</think>
<answer>2</answer>
        """
        ground_truth_format_error = "2"
        result_format_error = compute_score("thinklite_eureka",predict_str_format_error, ground_truth_format_error, extra_info={'question': '1+1=?'})
        print(result_format_error)
