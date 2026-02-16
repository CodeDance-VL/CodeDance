import argparse
import ast
import base64
import io
import json
import math
import multiprocessing
import os
import random
import re
import signal
import sys
import tempfile
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from openai import OpenAI
from PIL import Image
from tqdm import tqdm
from verl.utils.reward_score.llm_judge_qwen_tool_step import compute_score


DEFAULT_TIMEOUT_SECONDS = 30

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='qwen', help='Model name for result save')
    parser.add_argument('--api_key', type=str, default='EMPTY', help='API key or comma-separated list of API keys')
    parser.add_argument(
        '--api_url',
        type=str,
        default=(
            'http://127.0.0.1:18901/v1,http://127.0.0.1:18902/v1,'
            'http://127.0.0.1:18903/v1,http://127.0.0.1:18904/v1,'
            'http://127.0.0.1:18905/v1,http://127.0.0.1:18906/v1,'
            'http://127.0.0.1:18907/v1,http://127.0.0.1:18908/v1'
        ),
        help='API URL or comma-separated list of API URLs',
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='/Path/wemath_testmini.parquet',
        help='Path to the parquet file',
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='./save/',
        help='Path to save the results',
    )
    parser.add_argument('--eval_model_name', type=str, default=None, help='Model name for evaluation')
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--data_source', type=str, default='default', help='Data source type for compute_score function')
    return parser


def parse_args() -> argparse.Namespace:
    return build_arg_parser().parse_args()


def parse_comma_separated(value: str) -> List[str]:
    items = [item.strip() for item in value.split(",")]
    cleaned = [item for item in items if item]
    return cleaned if cleaned else [value]


instruction_prompt_system = [
    {
        "type": "text",
        "text": "You are a helpful assistant.\n\nSolve the following problem step by step. You may write Python code to assist with the user query. When an image is supplied, you can either use the preloaded PIL Image object `input_image` or access the image file directly via the **relative path** `'input_image.jpg'`.",
    }
]

INSTRUCTION_FOLLOWING = "Think step-by-step within <think></think>. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code should be complete scripts, including necessary imports. \nEach code snippet is wrapped with `<code>\n```python\ncode snippet\n```\n</code>`. You must provide your final answer in <answer></answer>."


@contextmanager
def capture_plt_show(instance_id: str):
    """Patch plt.show() to capture figures within a worker process."""
    import matplotlib.pyplot as plt

    original_show = plt.show
    captured_figures_list = []

    def _captured_show(*_args, **_kwargs):
        _ = (_args, _kwargs)
        import matplotlib.pyplot as plt
        current_fig = plt.gcf()
        if current_fig and current_fig.get_axes():
            try:
                try:
                    w_in, h_in = current_fig.get_size_inches()
                    dpi_val = current_fig.get_dpi() or 72
                    width_px, height_px = w_in * dpi_val, h_in * dpi_val
                    if min(width_px, height_px) == 0:
                        aspect_ratio = float('inf')
                    else:
                        aspect_ratio = max(width_px, height_px) / min(width_px, height_px)

                    if aspect_ratio > 200 or width_px < 28 or height_px < 28:
                        plt.close(current_fig)
                        return
                except Exception:
                    pass

                img_buffer = io.BytesIO()
                current_fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
                img_buffer.seek(0)
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                captured_figures_list.append(f"data:image/png;base64,{img_base64}")
                img_buffer.close()
            except Exception as e:
                print(
                    f"ERROR [capture_plt_show:{instance_id}]: Failed to save figure {current_fig.number}: {e}\n",
                    file=sys.stderr,
                )
            finally:
                plt.close(current_fig)
        return

    try:
        plt.show = _captured_show
        yield captured_figures_list
    finally:
        plt.show = original_show


@contextmanager
def capture_pil_show():
    """Patch PIL.Image.Image.show() to capture figures within a worker process."""
    captured_pil_figures = []
    original_pil_show = None

    try:
        from PIL import Image
        original_pil_show = Image.Image.show
    except ImportError:
        yield captured_pil_figures
        return

    def _captured_pil_show(self, *_args, **_kwargs):
        _ = (_args, _kwargs)
        try:
            pil_image = self

            width, height = pil_image.size
            aspect_ratio = max(height, width) / min(height, width)
            if aspect_ratio > 200:
                return
            if width < 56 or height < 56:
                scale = 112 / min(width, height)

                new_w = int(width * scale)
                new_h = int(height * scale)
                pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)

            img_buffer = io.BytesIO()
            save_format = 'PNG'

            if pil_image.mode not in ['RGB', 'RGBA']:
                pil_image = pil_image.convert('RGB')

            pil_image.save(img_buffer, format=save_format)
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            captured_pil_figures.append(f"data:image/png;base64,{img_base64}")
            img_buffer.close()
        except Exception as e:
            print(f"ERROR [capture_pil_show]: Failed to save figure via show(): {e}\n", file=sys.stderr)

    try:
        from PIL import Image
        Image.Image.show = _captured_pil_show
        yield captured_pil_figures
    finally:
        if original_pil_show:
            from PIL import Image
            Image.Image.show = original_pil_show


def execute_code_with_context(
    code: str,
    context: Dict[str, Any],
    temp_dir: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Execute code in a sandboxed context and capture output and images."""
    import matplotlib.pyplot as plt
    import builtins as _builtins
    
    def timeout_handler(_signum, _frame):
        del _signum, _frame
        raise TimeoutError(f"Code execution timed out after {timeout_seconds} seconds")
    
    class _SandboxExit(Exception):
        def __init__(self, code: int = 0):
            self.code = code
            super().__init__(f"Sandboxed exit with code {code}")
    
    def _patched_exit(code: int = 0):
        raise _SandboxExit(code)
    
    def _blocked_input(*_args, **_kwargs):
        raise RuntimeError("input() is disabled inside the sandbox.")
    
    orig_exit = getattr(_builtins, "exit", None)
    orig_quit = getattr(_builtins, "quit", None)
    orig_sys_exit = sys.exit
    orig_input = getattr(_builtins, "input", None)
    
    _builtins.exit = _patched_exit
    _builtins.quit = _patched_exit
    sys.exit = _patched_exit
    _builtins.input = _blocked_input
    
    old_sigalrm_handler = None
    if hasattr(signal, 'SIGALRM') and hasattr(signal, 'setitimer'):
        old_sigalrm_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, float(timeout_seconds))
    elif hasattr(signal, 'SIGALRM'):
        old_sigalrm_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_seconds))
    
    execution_locals = context
    execution_locals.update({"plt": plt, "__builtins__": __builtins__})
    execution_globals = execution_locals
    
    if 'input_image_path' in execution_locals and 'input_image' not in execution_locals:
        try:
            p = execution_locals.get('input_image_path')
            if os.path.exists(p):
                execution_locals['input_image'] = Image.open(p)
        except Exception as e:
            print(f"WARNING: failed to lazy-load input_image from '{p}': {e}", file=sys.stderr)
    
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    figures_base64 = []
    original_cwd = os.getcwd()
    try:
        os.chdir(temp_dir)
        
        with capture_plt_show("local") as captured_matplotlib_figures, \
                capture_pil_show() as captured_pil_figures, \
                redirect_stdout(stdout_buffer), \
                redirect_stderr(stderr_buffer):
            
            try:
                node = ast.parse(code, mode='exec')
                
                if node and node.body and isinstance(node.body[-1], ast.Expr):
                    if len(node.body) > 1:
                        module_body = ast.Module(body=node.body[:-1], type_ignores=[])
                        exec(compile(module_body, filename='<ast>', mode='exec'), execution_globals, execution_locals)
                    
                    last_expr_node = ast.Expression(body=node.body[-1].value)
                    result_val = eval(compile(last_expr_node, filename='<ast>', mode='eval'), execution_globals, execution_locals)
                    
                    if 'PIL' in str(type(result_val)):
                        try:
                            pil_image = result_val
                            width, height = pil_image.size
                            aspect_ratio = max(height, width) / min(height, width)
                            if aspect_ratio < 200:
                                if width < 56 or height < 56:
                                    scale = 112 / min(width, height)
                                    new_w = int(width * scale)
                                    new_h = int(height * scale)
                                    pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
                                
                                img_buffer = io.BytesIO()
                                save_format = 'JPEG'
                                
                                if pil_image.mode not in ['RGB', 'RGBA']:
                                    pil_image = pil_image.convert('RGB')
                                
                                pil_image.save(img_buffer, format=save_format)
                                img_buffer.seek(0)
                                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                                figures_base64.append(f"data:image/png;base64,{img_base64}")
                                img_buffer.close()
                        except Exception as e:
                            print(f"ERROR: Failed to display PIL Image: {e}\n", file=sys.stderr)
                    
                    elif 'numpy.ndarray' in str(type(result_val)):
                        try:
                            if result_val.ndim in [2, 3] and result_val.size > 1:
                                image_array = result_val
                                
                                if image_array.dtype != np.uint8:
                                    if np.issubdtype(image_array.dtype, np.floating):
                                        image_array = (image_array.clip(0, 1) * 255).astype(np.uint8)
                                    else:
                                        image_array = image_array.astype(np.uint8)
                                
                                if image_array.ndim == 3 and image_array.shape[2] == 3:
                                    image_array = image_array[:, :, ::-1]
                                
                                pil_image = Image.fromarray(image_array)
                                
                                img_buffer = io.BytesIO()
                                save_format = 'JPEG'
                                if pil_image.mode not in ['RGB', 'RGBA']:
                                    pil_image = pil_image.convert('RGB')
                                
                                pil_image.save(img_buffer, format=save_format)
                                img_buffer.seek(0)
                                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                                figures_base64.append(f"data:image/png;base64,{img_base64}")
                                img_buffer.close()
                            else:
                                print(result_val)
                        except Exception as e:
                            print(f"ERROR: Failed to display numpy.ndarray as Image: {e}\n", file=sys.stderr)
                    
                    elif result_val is not None:
                        print(result_val)
                else:
                    exec(code, execution_globals, execution_locals)
                
            except SyntaxError:
                exec(code, execution_globals, execution_locals)
            
            figures_base64.extend(captured_matplotlib_figures)
            figures_base64.extend(captured_pil_figures)
        
        result = {
            'success': True,
            'stdout': stdout_buffer.getvalue(),
            'stderr': stderr_buffer.getvalue(),
            'figures': figures_base64,
            'updated_context': execution_locals,
        }
        
    except _SandboxExit as e:
        result = {
            'success': True,
            'stdout': stdout_buffer.getvalue()
            + "\nNOTE: Code execution was halted by a call to exit() or quit(). Do not use exit() or quit() in your code.",
            'stderr': stderr_buffer.getvalue(),
            'figures': figures_base64,
            'updated_context': execution_locals,
        }
    except TimeoutError as e:
        result = {
            'success': False,
            'error': str(e),
            'error_type': 'TimeoutError',
            'stdout': stdout_buffer.getvalue(),
            'stderr': stderr_buffer.getvalue(),
            'figures': figures_base64,
        }
    except Exception as e:
        err_type = type(e).__name__
        user_friendly_error = f"{err_type}: {str(e)}"
        
        if isinstance(e, SyntaxError) and hasattr(e, 'lineno') and e.lineno is not None:
            user_friendly_error += f" (line {e.lineno})"
        stderr_output = stderr_buffer.getvalue()
        
        result = {
            'success': False,
            'error': user_friendly_error,
            'stdout': stdout_buffer.getvalue(),
            'stderr': stderr_output,
            'figures': figures_base64,
        }
    finally:
        if orig_exit:
            _builtins.exit = orig_exit
        if orig_quit:
            _builtins.quit = orig_quit
        if orig_sys_exit:
            sys.exit = orig_sys_exit
        if orig_input:
            _builtins.input = orig_input
        
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
            if old_sigalrm_handler:
                signal.signal(signal.SIGALRM, old_sigalrm_handler)
        
        os.chdir(original_cwd)
    
    return result

def _extract_python_code_from_response(response_text: str) -> Optional[str]:
    """Extract Python code from ```python blocks or <code> tags."""
    code_tag_pattern = r'<code>\s*```python\s*\n(.*?)\n```\s*</code>'
    code_tag_matches = re.findall(code_tag_pattern, response_text, re.DOTALL)
    if code_tag_matches:
        return code_tag_matches[0].strip()
    
    pattern = r'```python\s*\n(.*?)\n```'
    matches = re.findall(pattern, response_text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return None

def encode_pil_image_to_base64(pil_image: Image.Image) -> str:
    """Convert a PIL image to base64."""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


def process_image_from_bytes(image_data: Any) -> Image.Image:
    if hasattr(image_data, 'save'):
        return image_data
    return Image.open(io.BytesIO(image_data))

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def process_single(
    idx: int,
    df_data: pd.Series,
    api_keys: List[str],
    api_urls: List[str],
    eval_model_name: str,
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    selected_api_key = random.choice(api_keys)
    selected_api_url = random.choice(api_urls)
    
    local_client = OpenAI(
        api_key=selected_api_key,
        base_url=selected_api_url,
    )
    
    anno = df_data
    data_source = anno.get('data_source') or args.data_source
    image_data = anno.get('images')
    if image_data is not None:
        try:
            first_image = image_data[0]
            if isinstance(first_image, dict) and 'bytes' in first_image:
                if first_image['bytes'] is None or first_image['bytes'] == "":
                    img = Image.open(first_image['path'])
                else:
                    img = process_image_from_bytes(first_image['bytes'])
            else:
                img = process_image_from_bytes(image_data)
            if img.mode == 'P':
                if 'transparency' in img.info:
                    img = img.convert('RGBA')
                img = img.convert('RGB')
            elif img.mode == 'RGBA':
                img = img.convert('RGB')
        except Exception as e:
            print(f"Warning: failed to process image at index {idx} - {e}")
            return None
    else:
        print(f"Warning: missing image data at index {idx}")
        img = None

    if img is not None:    
        ori_width, ori_height = img.size
        resize_w, resize_h = smart_resize(ori_width, ori_height, factor=IMAGE_FACTOR)
        img = img.resize((resize_w, resize_h), resample=Image.BICUBIC)
        base64_image = encode_pil_image_to_base64(img)
        all_images = [base64_image]
        input_image = img
    else:
        all_images = []
        input_image = None
    
    question = anno['extra_info']['question'] 
    answer = anno['extra_info']['answer']
    prompt_content = (question + "\n" + INSTRUCTION_FOLLOWING).replace("<image>","")
    messages = [
        {
            "role": "system",
            "content": instruction_prompt_system,
        },
        {
            "role": "user",
            "content": [
                *([{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, "max_pixels": MAX_PIXELS}] if img is not None else []),
                {"type": "text", "text": prompt_content},
            ],
        }
    ]
    print_messages = [
        {
            "role": "system",
            "content": instruction_prompt_system,
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"}, "max_pixels": MAX_PIXELS,} if img is not None else {},
                {"type": "text", "text": prompt_content},
            ],
        }
    ]

    chat_message = messages

    response_message = ""

    status = 'success'
    try_count = 0
    tool_call_count = 0
    tool_success_count = 0
    
    try:
        with tempfile.TemporaryDirectory(prefix="code_exec_") as temp_dir:
            execution_context: Dict[str, Any] = {}
            if input_image is not None:
                image_path = os.path.join(temp_dir, "input_image.jpg")
                input_image.save(image_path)
                execution_context['input_image_path'] = image_path
                execution_context['input_image'] = input_image

            while True:
                if '</answer>' in response_message and '<answer>' in response_message:
                    break
                if try_count > 6:
                    break

                params = {
                    "model": eval_model_name,
                    "messages": chat_message,
                    "temperature": 0.01,
                    "max_tokens": 2048,
                    "top_p": 1,
                }
                response = local_client.chat.completions.create(**params)
                response_message = response.choices[0].message.content
                python_code = _extract_python_code_from_response(response_message)

                if python_code:
                    tool_call_count += 1
                    try:
                        result = execute_code_with_context(
                            code=python_code,
                            context=execution_context,
                            temp_dir=temp_dir,
                            timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
                        )
                        if result.get('success') and 'updated_context' in result:
                            execution_context = result['updated_context']

                        tool_response_content = [{"type": "text", "text": "<tool_response>"}]

                        stdout_output = result.get('stdout', '').strip()
                        if stdout_output:
                            tool_response_content.append({"type": "text", "text": f"Output:\n{stdout_output}"})

                        stderr_output = result.get('stderr', '').strip()
                        if stderr_output:
                            tool_response_content.append(
                                {"type": "text", "text": f"Stderr output:\n```\n{stderr_output}\n```"}
                            )

                        figures = result.get('figures', [])
                        for img_base64 in figures:
                            tool_response_content.append({"type": "text", "text": "Output image:"})
                            tool_response_content.append({"type": "image_url", "image_url": {"url": img_base64}})
                            all_images.append(img_base64)

                        if not result.get('success'):
                            error_msg = result.get('error', 'Unknown execution error')
                            error_type = result.get('error_type', 'Error')
                            full_error = f"--- Execution Error ({error_type}) ---\n{error_msg}"
                            tool_response_content.append({"type": "text", "text": full_error})

                        tool_response_content.append({"type": "text", "text": "</tool_response>"})

                        error_strings = [
                            "Code executed successfully. No explicit output or figures produced.",
                            "Error",
                            "timeout",
                            "Exception",
                        ]
                        error_strings_lower = [s.lower() for s in error_strings]
                        tool_response_text = "\n".join(
                            item.get("text", "")
                            for item in tool_response_content
                            if isinstance(item, dict) and item.get("type") == "text"
                        )
                        if all(s not in tool_response_text.lower() for s in error_strings_lower) and not stderr_output:
                            tool_success_count += 1
                        else:
                            print(f"Tool execution failed: {tool_response_text}")

                        chat_message.append({
                            "role": "assistant",
                            "content": response_message,
                        })

                        chat_message.append({
                            "role": "user",
                            "content": tool_response_content,
                        })

                        print_messages.append({
                            "role": "assistant",
                            "content": response_message,
                        })

                        tool_response_content_print = [{"type": "text", "text": "<tool_response>"}]
                        tool_response_content_print.extend(tool_response_content)
                        tool_response_content_print.append({"type": "text", "text": "</tool_response>"})

                        print_messages.append({
                            "role": "user",
                            "content": tool_response_content_print,
                        })
                    except Exception as tool_error:
                        print(f"Tool execution error: {tool_error}")
                        error_msg = f"Code execution failed: {str(tool_error)}"
                        chat_message.append({
                            "role": "assistant",
                            "content": response_message,
                        })
                        chat_message.append({
                            "role": "user",
                            "content": [{"type": "text", "text": error_msg}],
                        })

                        print_messages.append({
                            "role": "assistant",
                            "content": response_message,
                        })
                        print_messages.append({
                            "role": "user",
                            "content": [{"type": "text", "text": error_msg}],
                        })
                else:
                    print_messages.append({
                        "role": "assistant",
                        "content": response_message,
                    })
                try_count += 1
    except Exception as e:
        print(f"Error: {e}")
        status = 'error'
    output_text = response_message
    try:
        judge_result = compute_score(
            data_source=data_source,
            solution_str=output_text,
            ground_truth=answer,
            extra_info={'question': question}
        )
        acc_score = judge_result.get('acc', 0.0)
        judge_score = judge_result.get('score', 0.0)
    except Exception as e:
        print(f"Judge model error at index {idx}: {e}")
        acc_score = 0.0
        judge_score = 0.0
        judge_result = {}
    save_info = {}
    save_info['question'] = question
    save_info['ground_truth'] = answer
    save_info['pred_ans'] = output_text
    
    conversation_string = ""
    if print_messages:
        for msg in print_messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                text_content = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_content.append(item.get("text", ""))
                msg["content"] = "\n".join(text_content)
        
        conversation_string = "\n".join([f"{msg['role']}\n{msg['content']}" for msg in print_messages])

    save_info['input'] = ""
    save_info['output'] = conversation_string
    save_info['multi_modal_data'] = {'image': all_images}
    
    save_info['status'] = status
    save_info['index'] = idx
    save_info['acc'] = acc_score
    save_info['judge_score'] = judge_score
    save_info['judge_result'] = judge_result
    save_info['try_count'] = try_count
    save_info['tool_call_count'] = tool_call_count
    save_info['tool_success_count'] = tool_success_count
    save_info['tool_success_rate'] = tool_success_count / tool_call_count if tool_call_count > 0 else None

    return save_info

def worker_process(task_data):
    idx, df_row, api_keys, api_urls, eval_model_name, args = task_data
    try:
        return process_single(idx, df_row, api_keys, api_urls, eval_model_name, args)
    except Exception as e:
        print(f"Error processing index {idx}: {e}")
        return None

def main(
    args: argparse.Namespace,
    api_keys: List[str],
    api_urls: List[str],
    eval_model_name: str,
) -> List[Dict[str, Any]]:
    save_json = []
    df = pd.read_parquet(args.data_path)
    rows_len = df.shape[0]
    
    tasks = []
    for idx in range(rows_len):
        df_row = df.iloc[idx]
        tasks.append((idx, df_row, api_keys, api_urls, eval_model_name, args))
    
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        with tqdm(total=rows_len, desc=f"Processing dataset: ") as pbar:
            for result in pool.imap(worker_process, tasks):
                if result is not None:
                    save_json.append(result)
                pbar.update(1)

    return save_json


def resolve_eval_model_name(api_base: str, eval_model_name: Optional[str]) -> str:
    if eval_model_name is None:
        response = requests.get(f"{api_base}/models")
        response.raise_for_status()
        models = response.json()
        return models['data'][0]['id']
    return eval_model_name


def run(args: argparse.Namespace) -> None:
    api_keys = parse_comma_separated(args.api_key)
    api_urls = parse_comma_separated(args.api_url)

    eval_model_name = resolve_eval_model_name(api_urls[0], args.eval_model_name)

    save_path = os.path.join(args.save_path, args.model_name)
    os.makedirs(save_path, exist_ok=True)

    save_json = main(args, api_keys, api_urls, eval_model_name)

    data_name = os.path.splitext(os.path.basename(args.data_path))[0]
    save_name = f"result_{args.model_name}_{data_name}.jsonl"

    total_samples = len(save_json)
    total_acc = sum(item.get('acc', 0.0) for item in save_json)
    avg_acc = total_acc / total_samples if total_samples > 0 else 0.0

    total_judge_score = sum(item.get('judge_score', 0.0) for item in save_json)
    avg_judge_score = total_judge_score / total_samples if total_samples > 0 else 0.0

    total_try_count = sum(item.get('try_count', 0) for item in save_json)
    avg_try_count = total_try_count / total_samples if total_samples > 0 else 0.0
    avg_reasoning_turns = avg_try_count

    success_samples = sum(1 for item in save_json if item.get('status') == 'success')
    success_rate = success_samples / total_samples if total_samples > 0 else 0.0

    tool_call_samples = [item for item in save_json if item.get('tool_call_count', 0) > 0]
    tool_call_sample_count = len(tool_call_samples)
    total_tool_success_rate = sum(item.get('tool_success_rate', 0.0) for item in tool_call_samples)
    avg_code_exec_success_rate = (
        total_tool_success_rate / tool_call_sample_count if tool_call_sample_count > 0 else 0.0
    )

    try_count_stats: Dict[int, Dict[str, int]] = {}
    for item in save_json:
        count = item.get('try_count', 0)
        if count not in try_count_stats:
            try_count_stats[count] = {'correct': 0, 'incorrect': 0}

        if item.get('acc', 0.0) == 1.0:
            try_count_stats[count]['correct'] += 1
        else:
            try_count_stats[count]['incorrect'] += 1

    print("\n=== Samples by try count ===")
    for count, stats in sorted(try_count_stats.items()):
        print(f"Try count = {count}:")
        print(f"  Correct samples: {stats['correct']}")
        print(f"  Incorrect samples: {stats['incorrect']}")

    print("\n=== Evaluation summary ===")
    print(f"Total samples: {total_samples}")
    print(f"Successful samples: {success_samples}")
    print(f"Success rate: {success_rate:.4f}")
    print(f"Average accuracy (ACC): {avg_acc:.4f}")
    print(f"Total correct samples: {total_acc}")
    print(f"Average judge score: {avg_judge_score:.4f}")
    print(f"Average try count: {avg_try_count:.4f}")
    print(f"Average reasoning turns: {avg_reasoning_turns:.4f}")
    print(f"Samples with code execution: {tool_call_sample_count}")
    print(f"Average code execution success rate: {avg_code_exec_success_rate:.4f}")

    stats = {
        'total_samples': total_samples,
        'success_samples': success_samples,
        'success_rate': success_rate,
        'avg_acc': avg_acc,
        'avg_judge_score': avg_judge_score,
        'avg_try_count': avg_try_count,
        'avg_reasoning_turns': avg_reasoning_turns,
        'tool_call_sample_count': tool_call_sample_count,
        'avg_code_exec_success_rate': avg_code_exec_success_rate,
        'model_name': args.model_name,
        'data_source': args.data_source,
    }

    save_json_new = [item for item in save_json if item.get('try_count', 0) > 2]
    with open(os.path.join(save_path, save_name), 'w') as f:
        for item in save_json_new:
            f.write(json.dumps(item) + '\n')

    stats_name = f"stats_{args.model_name}.json"
    with open(os.path.join(save_path, stats_name), 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    run(parse_args())
