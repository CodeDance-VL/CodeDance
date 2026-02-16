import logging
import os
import io
import base64
import shutil
import asyncio
import ast
import traceback
import sys
import threading
import signal
import multiprocessing
from multiprocessing import Process, Queue
import pickle
from uuid import uuid4
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from typing import Any, Optional, Tuple, Dict, List, Union
import gc
from PIL import Image
import types
import numpy as np
import time
# --- Assume these imports are available in the environment ---
from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema
# --- End Mock ---
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from verl.utils.dataset.vision_utils import process_image

# --- Logging setup ---
logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

# Constants
DEFAULT_TIMEOUT_SECONDS = 3 # Increased default for potentially heavier tasks
DEFAULT_INIT_TIMEOUT_SECONDS = 30  # Timeout for the initial "initialize" handshake with a worker
TEMP_DIR_BASE = "/tmp/code_execute_tool_instances_pool"
MAX_WORKERS = 32 #int(os.cpu_count() // 4 or 1)
FRAMEWORK_COMM_TIMEOUT = 120 # 3 minutes, for detecting unresponsive workers.
USER_PROMPT = "\nThink with the tool's result, then answer. Determine if it adequately answers the user's question. If the information is insufficient or irrelevant, formulate a new strategy and execute a new tool call to obtain a helpful result."


@contextmanager
def capture_plt_show(instance_id: str):
    """
    Context manager to temporarily patch plt.show(), capture figures,
    and store them in thread-local storage. It closes only the captured figure.
    This needs to be used *within* the worker process.
    """
    import matplotlib.pyplot as plt
    
    log_extra = {'instance_id': instance_id}
    original_show = plt.show
    captured_figures_list = []

    def _captured_show(*args, **kwargs):
        import matplotlib.pyplot as plt
        current_fig = plt.gcf()
        if current_fig and current_fig.get_axes():
            try:
                # Skip capturing figures with an extreme aspect-ratio to avoid
                # accidentally returning huge, unreadable images.
                # "Aspect-ratio" here is defined as:  max(width, height) / min(width, height)
                # (both measured in pixels).  If this value exceeds 200 we drop the figure.
                try:
                    w_in, h_in = current_fig.get_size_inches()
                    dpi_val = current_fig.get_dpi() or 72
                    width_px, height_px = w_in * dpi_val, h_in * dpi_val
                    # Guard against division-by-zero for degenerate figures.
                    if min(width_px, height_px) == 0:
                        aspect_ratio = float('inf')
                    else:
                        aspect_ratio = max(width_px, height_px) / min(width_px, height_px)

                    if aspect_ratio > 200 or width_px<28 or height_px < 28:
                        # Do **not** output this figure, just close and return.
                        plt.close(current_fig)
                        return
                except Exception:
                    # If the size check itself fails, default to capturing the image –
                    # we prefer a potential false-positive over silencing legitimate output.
                    pass

                img_buffer = io.BytesIO()
                current_fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
                img_buffer.seek(0)
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                captured_figures_list.append(f"data:image/png;base64,{img_base64}")
                img_buffer.close()
            except Exception as e:
                print(f"ERROR [capture_plt_show:{instance_id}]: Failed to save figure {current_fig.number}: {e}\n", file=sys.stderr)
            finally:
                plt.close(current_fig)
        else:
            pass

    try:
        plt.show = _captured_show
        yield captured_figures_list
    finally:
        plt.show = original_show

@contextmanager
def capture_pil_show():
    """
    Context manager to temporarily patch PIL.Image.Image.show(), capture figures,
    and store them. This needs to be used *within* the worker process.
    """
    captured_pil_figures = []
    original_pil_show = None

    try:
        # Import dynamically to avoid issues if PIL isn't installed in the main env.
        from PIL import Image
        original_pil_show = Image.Image.show
    except ImportError:
        # If PIL is not available, this context manager does nothing.
        yield captured_pil_figures
        return

    def _captured_pil_show(self, *args, **kwargs):
        """Replacement for PIL.Image.Image.show()."""
        try:
            pil_image = self  # 'self' is the Image instance
            
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
            save_format = 'PNG'  # Use PNG to preserve transparency
            
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
        # Restore the original function if it was patched.
        if original_pil_show:
            from PIL import Image
            Image.Image.show = original_pil_show

# --- Worker Function (Runs in Process Pool) ---
def execute_code_in_worker(code: str,
                           instance_id: str,
                           current_context: Dict[str, Any],
                           temp_dir: str,
                           timeout_seconds: float) -> Dict[str, Any]:
    """
    Executes code within a worker process. User code handles necessary imports.
    Context (locals) persists across calls for the same instance.
    Handles I/O redirection, plotting (via pre-imported plt), and timeout.
    Returns results including the filtered, serializable updated execution context.
    """
    log_prefix = f"[execute_code_in_worker:{instance_id}]"
    import matplotlib.pyplot as plt

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Code execution timed out after {timeout_seconds} seconds")

    old_handler = None
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout_seconds))
    else:
        print(f"WARNING {log_prefix}: signal.SIGALRM not available. Timeout may not be enforced.", file=sys.stderr)

    # Prevent user code from tampering with our alarm or signal handler
    _orig_alarm_fn = getattr(signal, 'alarm', None)
    _orig_signal_fn = getattr(signal, 'signal', None)  # Keep for later restore, but don't patch.

    # Block user from scheduling new alarms while allowing the framework itself to manage SIGALRM.
    def _blocked_alarm(*_a, **_kw):
        raise RuntimeError("signal.alarm() is disabled inside the sandbox.")

    if _orig_alarm_fn:
        signal.alarm = _blocked_alarm  # type: ignore

    # NOTE: We intentionally do NOT monkey-patch `signal.signal` here.  Doing so caused
    #       the framework's own subsequent call to `signal.signal()` (to install the
    #       timeout handler on the next execution) to raise a RuntimeError once the
    #       original function was accidentally lost.  By leaving `signal.signal`
    #       untouched we retain full control over the timeout mechanism while still
    #       preventing user code from extending the alarm duration.


    execution_locals = current_context  # keep the very same dict for persistence
    execution_locals.update({
        "plt": plt,
        "__builtins__": __builtins__,
    })

    # Re-use the same reference for globals to ensure a single coherent namespace.
    execution_globals = execution_locals

    # --- Sandbox patches to prevent blocking input() or premature process termination ---
    import builtins as _builtins

    class _SandboxExit(Exception):
        """Raised when user code attempts to exit the interpreter (via exit/quit/sys.exit)."""
        def __init__(self, code: int = 0):
            self.code = code
            super().__init__(f"Sandboxed exit with code {code}")

    def _patched_exit(code: int = 0):
        # We raise a custom exception that is caught later so that the worker process stays alive.
        raise _SandboxExit(code)

    # --- Temporarily patch exit / quit / sys.exit ---------------------------
    # We *must* restore the originals in the `finally` block, otherwise the
    # worker process itself (or the multiprocessing runtime) could run into
    # our stub when it legitimately calls `sys.exit()`, leading to tracebacks
    # like the one reported by the user.
    orig_exit = getattr(_builtins, "exit", None)
    orig_quit = getattr(_builtins, "quit", None)
    orig_sys_exit = sys.exit

    _builtins.exit = _patched_exit  # type: ignore
    _builtins.quit = _patched_exit  # type: ignore
    sys.exit = _patched_exit  # type: ignore
    # -------------------------------------------------------------------------

    if 'input_image_path' in execution_locals and 'input_image' not in execution_locals:
        try:
            from PIL import Image  # local import – PIL is already available in the worker
            p = execution_locals.get('input_image_path')
            if os.path.exists(p):
                execution_locals['input_image'] = Image.open(p)
        except Exception as e:
            print(f"WARNING: failed to lazy-load input_image from '{p}': {e}", file=sys.stderr)

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    figures_base64 = []
    result = {}

    original_cwd = os.getcwd()
    try:
        os.chdir(temp_dir)
        
        with capture_plt_show(instance_id) as captured_matplotlib_figures, \
             capture_pil_show() as captured_pil_figures, \
             redirect_stdout(stdout_buffer), \
             redirect_stderr(stderr_buffer):

            try:
                # Attempt to parse the code into an abstract syntax tree.
                node = ast.parse(code, mode='exec')

                # If the last node is an expression, we want to print its result.
                if node and node.body and isinstance(node.body[-1], ast.Expr):
                    
                    # Execute all statements except for the last one.
                    if len(node.body) > 1:
                        module_body = ast.Module(body=node.body[:-1], type_ignores=[])
                        exec(compile(module_body, filename='<ast>', mode='exec'), execution_globals, execution_locals)
                    
                    # Evaluate the last expression and store its result.
                    last_expr_node = ast.Expression(body=node.body[-1].value)
                    result_val = eval(compile(last_expr_node, filename='<ast>', mode='eval'), execution_globals, execution_locals)

                    # Check if the result is a PIL-like image by inspecting its type name string.
                    # This is the most robust method in this execution context.
                    if 'PIL' in str(type(result_val)):
                        try:
                            from PIL import Image
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
                    
                    # Handle numpy arrays, which are common for cv2 images.
                    elif 'numpy.ndarray' in str(type(result_val)):
                        try:
                            from PIL import Image
                            # Heuristic: check if the array looks like an image (2D or 3D).
                            if result_val.ndim in [2, 3] and result_val.size > 1:
                                image_array = result_val

                                # Ensure the data type is uint8 for PIL.
                                # This handles float arrays (e.g., 0.0-1.0) common in processing.
                                if image_array.dtype != np.uint8:
                                    if np.issubdtype(image_array.dtype, np.floating):
                                        image_array = (image_array.clip(0, 1) * 255).astype(np.uint8)
                                    else:
                                        image_array = image_array.astype(np.uint8)

                                # Convert BGR (OpenCV default) to RGB (PIL default) for 3-channel images.
                                if image_array.ndim == 3 and image_array.shape[2] == 3:
                                    image_array = image_array[:, :, ::-1]
                                
                                pil_image = Image.fromarray(image_array)

                                # --- Re-use PIL saving logic ---
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
                                # Not an image-like array, just print it.
                                print(result_val)
                        except Exception as e:
                            print(f"ERROR: Failed to display numpy.ndarray as Image: {e}\n", file=sys.stderr)

                    # For other types, print the result to stdout if it's not None. and print result_val name.
                    elif result_val is not None:
                        print(result_val)
                else:
                    # If it's not an expression, execute the whole code block as a script.
                    exec(code, execution_globals, execution_locals)
            
            except SyntaxError:
                # If parsing fails, fall back to `exec` so the original error
                # handling can report it consistently.
                exec(code, execution_globals, execution_locals)
            
            figures_base64.extend(captured_matplotlib_figures)
            figures_base64.extend(captured_pil_figures)

        # With persistent workers, we no longer need to serialize the context.
        # The 'execution_locals' dictionary *is* the new context, containing
        # all variables, functions, and imports. We return it directly.
        updated_locals = execution_locals

        result = {
            'success': True,
            'stdout': stdout_buffer.getvalue(),
            'stderr': stderr_buffer.getvalue(),
            'figures': figures_base64,
            'updated_locals': updated_locals
        }

    except _SandboxExit as e:
        # This indicates user code called exit()/quit(). Treat as success with a note.
        # This makes the behavior more explicit and prevents it from being logged as a generic error.
        result = {
            'success': True,
            'stdout': stdout_buffer.getvalue() + "\nNOTE: Code execution was halted by a call to exit() or quit(). Do not use exit() or quit() in your code.",
            'stderr': stderr_buffer.getvalue(),
            'figures': figures_base64,
            'updated_locals': execution_locals
        }
    except TimeoutError as e:
        logger.warning(f"{log_prefix} Execution timed out: {e}, code: {code}", extra={'instance_id': instance_id})
        result = {
            'success': False, 'error': str(e), 'error_type': 'TimeoutError',
            'stdout': stdout_buffer.getvalue(), 'stderr': stderr_buffer.getvalue(),
            'figures': figures_base64
        }
    except Exception as e:
        err_type = type(e).__name__
        user_friendly_error = f"{err_type}: {str(e)}"
        
        if isinstance(e, SyntaxError) and hasattr(e, 'lineno') and e.lineno is not None:
             user_friendly_error += f" (line {e.lineno})"
        stderr_output = stderr_buffer.getvalue()

        result = {
            'success': False, 'error': user_friendly_error,
            'stdout': stdout_buffer.getvalue(), 'stderr': stderr_output,
            'figures': figures_base64
        }
    finally:
        # Restore signal functions patched earlier BEFORE we cancel the alarm to avoid invoking the blocked stub.
        if _orig_alarm_fn:
            signal.alarm = _orig_alarm_fn  # type: ignore
        if _orig_signal_fn:
            signal.signal = _orig_signal_fn  # type: ignore

        if hasattr(signal, 'SIGALRM') and old_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        os.chdir(original_cwd)
        # Restore signal functions patched earlier BEFORE we cancel the alarm to avoid invoking the blocked stub.

        if 'orig_exit' in locals():
            if orig_exit is not None:
                _builtins.exit = orig_exit  # type: ignore
            if orig_quit is not None:
                _builtins.quit = orig_quit  # type: ignore
            sys.exit = orig_sys_exit  # type: ignore

        plt.close('all')
        gc.collect()

    return result

def worker_main_loop(task_queue: Queue, result_queue: Queue):
    """The main loop for a worker process. Manages contexts for multiple instances."""
    # Set a non-interactive backend for matplotlib BEFORE it's ever imported by user code.
    # This is crucial to prevent hangs in subprocesses when plotting.
    import matplotlib
    matplotlib.use('Agg')
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '' # Disable CUDA for this worker process
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    import cv2  # noqa: F401 – optional dependency, may not be installed

    def _cv2_gui_disabled(*_args, **_kwargs):  # type: ignore
        """Stub for GUI functions in cv2 that would normally open a window."""
        raise RuntimeError(
            "cv2 GUI functions (e.g. namedWindow / waitKey / destroyAllWindows) "
            "are disabled inside the execution sandbox. Use matplotlib or PIL to display "
            "images instead."
        )
    _cv2_gui_func_names = [
        "imshow", "imshowMulti", "namedWindow", "startWindowThread", "waitKey",
        "waitKeyEx", "pollKey", "destroyAllWindows", "destroyWindow",
        "createTrackbar", "getTrackbarPos", "setTrackbarPos", "setTrackbarMin",
        "setTrackbarMax", "selectROI", "selectROIs",
    ]
    for _fn in _cv2_gui_func_names:
        if hasattr(cv2, _fn):
            setattr(cv2, _fn, _cv2_gui_disabled)
    # -----------------------------------------------------------------------------

    instance_contexts = {}
    #logger.info(f"Worker process {os.getpid()} started.")

    while True:
        try:
            task = task_queue.get()
            if task is None:
                logger.info(f"Worker process {os.getpid()} shutting down.")
                break

            request_id, command, payload = task
            instance_id = payload.get('instance_id')

            if command == 'execute':
                if instance_id not in instance_contexts:
                    # This is the first time this worker has seen this instance_id.
                    # This is the normal and expected behavior; the warning was incorrect.
                    instance_contexts[instance_id] = {'context': {}}
                
                current_context = instance_contexts[instance_id]['context']
                # Inject the image path into the execution context on the first call so
                # that `execute_code_in_worker` can lazily load it into an `input_image`
                # object before running user code.
                image_path = payload.get('image_path')
                if image_path and 'input_image_path' not in current_context and 'input_image' not in current_context:
                    current_context['input_image_path'] = image_path
                result = execute_code_in_worker(
                    code=payload['code'],
                    instance_id=instance_id,
                    current_context=current_context,
                    temp_dir=payload['temp_dir'],
                    timeout_seconds=payload['timeout_seconds']
                )

                if result.get('success'):
                    instance_contexts[instance_id]['context'] = result.pop('updated_locals', {})
                else:
                    instance_contexts[instance_id]['context'] = {}
                
                result_queue.put((request_id, result))

            elif command == 'release':
                if instance_id in instance_contexts:
                    instance_contexts[instance_id].get('context', {}).clear()
                    del instance_contexts[instance_id]
                    gc.collect()
                # send ack back if request_id is not None
                if request_id is not None:
                    result_queue.put((request_id, {"success": True}))
        
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Worker process {os.getpid()} caught exception: {e}\n{tb}")
            if 'request_id' in locals() and request_id:
                result_queue.put((request_id, {
                    'success': False, 'error': f"Worker process failed: {e}",
                    'error_type': 'WorkerError', 'stdout': '', 'stderr': tb, 'figures': []
                }))
        finally:
            gc.collect()

# --- Main Tool Class ---
class CodeExecuteTool(BaseTool):
    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        super().__init__(config, tool_schema)
        self._instance_dict: Dict[str, Dict[str, Any]] = {}
        
        self.max_workers = config.get("max_workers", MAX_WORKERS)

        image_save_max_workers = int(config.get("image_save_max_workers", min(32, self.max_workers * 4)))
        # NOTE: instantiating our own pool lets us cap the thread count instead
        # of relying on asyncio's default, which might be too small (often 5).
        self.image_save_executor = ThreadPoolExecutor(max_workers=image_save_max_workers)
        self.task_queues: List[Queue] = []
        self.workers: List[Process] = []
        self.result_queue = Queue()
        self.pending_futures: Dict[str, asyncio.Future] = {}
        self._next_worker_idx = 0
        self._worker_assignment_lock = asyncio.Lock()
        self._result_reader_task: Optional[asyncio.Task] = None
        os.makedirs(TEMP_DIR_BASE, exist_ok=True)

        for i in range(self.max_workers):
            task_queue = Queue()
            worker = Process(
                target=worker_main_loop,
                args=(task_queue, self.result_queue),
                name=f"CodeExecWorker-{i}"
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            self.task_queues.append(task_queue)
        
    async def _read_results_loop(self):
        #logger.info("Result reader loop started.")
        loop = asyncio.get_running_loop()
        while True:
            try:
                # The previous implementation used `self.result_queue.empty()`, which is unreliable
                # for multiprocessing queues and introduces a polling behavior with a sleep.
                # A blocking `get()` in an executor thread is more efficient and responsive. It
                # waits for a result to be available without busy-waiting.
                request_id, result = await loop.run_in_executor(None, self.result_queue.get)

                if request_id in self.pending_futures:
                    future = self.pending_futures.pop(request_id)
                    # Add a check to prevent setting result on a future that's already done (e.g., cancelled by timeout)
                    if not future.done():
                        future.set_result(result)
                else:
                    logger.warning(f"Received result for unknown/timed-out request_id: {request_id}")
            except (asyncio.CancelledError, EOFError, BrokenPipeError):
                logger.info("Result reader loop is shutting down.")
                break
            except Exception as e:
                logger.error(f"Error in result reader loop: {e}", exc_info=True)
                for future in self.pending_futures.values():
                    if not future.done():
                        future.set_exception(e)
                self.pending_futures.clear()
                break

    async def _send_request(self, worker_id: int, command: str, payload: Dict[str, Any], timeout: Optional[float] = None) -> Any:
        # Lazily start the result-reader on **this** loop if it hasn't been started yet.
        loop = asyncio.get_running_loop()
        if self._result_reader_task is None:
            self._result_reader_task = loop.create_task(self._read_results_loop())

        request_id = str(uuid4())
        future = loop.create_future()
        self.pending_futures[request_id] = future

        task = (request_id, command, payload)
        self.task_queues[worker_id].put(task)
        
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self.pending_futures.pop(request_id, None)
            raise

    async def create(self, instance_id: Optional[str] = None, image: Optional[Union[Image.Image, str, dict]] = None, **kwargs) -> str:
        """Creates a new execution instance and assigns it to a worker."""
        if instance_id is None:
            instance_id = str(uuid4())
        log_extra = {'instance_id': instance_id}
        if instance_id in self._instance_dict:
            logger.warning(f"Instance '{instance_id}' already exists. Releasing old one first.", extra=log_extra)
            await self.release(instance_id)

        async with self._worker_assignment_lock:
            worker_id = self._next_worker_idx % self.max_workers
            self._next_worker_idx += 1

        temp_dir = os.path.join(TEMP_DIR_BASE, instance_id)
        os.makedirs(temp_dir, exist_ok=True)
        self._instance_dict[instance_id] = {"worker_id": worker_id, "temp_dir": temp_dir}

        image_path: Optional[str] = None
        if image is not None:

            loop = asyncio.get_running_loop()
            image_path = await loop.run_in_executor(
                self.image_save_executor,
                _save_image_in_main_process,
                instance_id,
                image,
                temp_dir,
            )
            if not image_path:
                logger.warning(
                    f"Image for instance '{instance_id}' could not be saved; proceeding without it.",
                    extra=log_extra,
                )
        
        # Store extra metadata so execute() can forward it.
        self._instance_dict[instance_id]["image_path"] = image_path

        #logger.warning(f"Instance '{instance_id}' created and assigned to worker {worker_id}.", extra=log_extra)

        return instance_id

    async def execute(self, instance_id: str, parameters: Dict[str, Any], **kwargs) -> Tuple[Dict[str, List[Dict[str, str]]], float, Dict[str, Any]]:
        """Executes code for an instance in its assigned worker process."""
        log_extra = {'instance_id': instance_id}

        if instance_id not in self._instance_dict:
            error_msg = f"Error: Instance '{instance_id}' not found. Please create it first."
            return {"content": [{"type": "text", "text": error_msg}]}, 0.0, {"execution_successful": False, "error": "Instance not found"}

        instance_data = self._instance_dict[instance_id]
        worker_id = instance_data["worker_id"]
        temp_dir = instance_data["temp_dir"]

        code = parameters.get("code", "")
        if not isinstance(code, str) or not code.strip():
            error_msg = "Error: No code provided to execute."
            return {"content": [{"type": "text", "text": error_msg}]}, 0.0, {"execution_successful": False, "error": "No code provided"}

        try:
            timeout_seconds = float(parameters.get("timeout", os.getenv("CODE_EXECUTION_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS))))
        except (ValueError, TypeError):
            timeout_seconds = DEFAULT_TIMEOUT_SECONDS
        if timeout_seconds <= 0:
            timeout_seconds = DEFAULT_TIMEOUT_SECONDS

        result_content: List[Dict[str, str]] = []
        metadata: Dict[str, Any] = {"execution_successful": False}
        reward = -0.5

        try:
            #logger.info(f"Sending code to worker {worker_id} for instance '{instance_id}' (timeout={timeout_seconds}s)...", extra=log_extra)
            # The user-provided timeout_seconds is sent to the worker to enforce a strict
            # limit on code execution.
            payload = {
                'instance_id': instance_id,
                'code': code,
                'temp_dir': temp_dir,
                'timeout_seconds': timeout_seconds,
                'image_path': instance_data.get('image_path'),
            }
            
            # We now use a separate, long, fixed timeout for framework communication.
            # This is a safety net to detect a completely hung or crashed worker process,
            # and it is decoupled from the user's code execution timeout.
            result = await self._send_request(worker_id, 'execute', payload, timeout=FRAMEWORK_COMM_TIMEOUT)
            
            stdout_output = result.get('stdout', '')
            stderr_output = result.get('stderr', '')
            displayed_figures = result.get('figures', [])

            if result.get('success'):
                if stdout_output.strip():
                    result_content.append({"type": "text", "text": f"Code execution successful, output:\n{stdout_output.strip()}"})
                if stderr_output.strip():
                     result_content.append({"type": "text", "text": f"Stderr output:\n```\n{stderr_output.strip()}\n```"})
                
                for img_base64 in displayed_figures:
                    result_content.append({"type": "text", "text": f"Output image:"})
                    result_content.append({"type": "image", "image": img_base64})
                result_content.append({"type": "text", "text": USER_PROMPT})

                metadata["execution_successful"] = True
                reward = 1.0
                if not result_content:
                    result_content.append({"type": "text", "text": "Code executed successfully. No explicit output or figures produced. You should use the `print` function to output text. Rewrite the code."})

            else: # Execution failed in worker
                error_msg = result.get('error', 'Unknown execution error')
                error_type = result.get('error_type', 'Error')
                if stdout_output.strip():
                    result_content.append({"type": "text", "text": f"Output before error:\n{stdout_output.strip()}"})
                full_error = f"--- Execution Error ({error_type}) ---\n{error_msg}"
                if stderr_output.strip():
                    full_error += f"\n\nStderr output:\n{stderr_output.strip()}"
                result_content.append({"type": "text", "text": full_error})
                metadata["error"] = error_type
                metadata["error_details"] = error_msg

        except asyncio.TimeoutError:
            # This timeout occurs if the worker process becomes unresponsive. This can happen
            # if the executed code causes a deadlock, an infinite loop that cannot be interrupted,
            # or if it calls a blocking function like `breakpoint()` or `input()`.
            error_msg_detail = (
                f"Code execution timed out."
            )
            logger.error(f"Framework timeout for instance '{instance_id}'. Code:\n {code}", extra=log_extra)
            result_content.append({"type": "text", "text": error_msg_detail})
            # Classify as a user-level timeout, not a framework failure.
            metadata["error"] = "TimeoutError"
            metadata["error_details"] = error_msg_detail

        except Exception as e:
            error_msg_detail = f"An unexpected exception': {type(e).__name__}: {str(e)}"
            logger.exception(f"Unexpected framework error for instance '{instance_id}'.", extra=log_extra)
            result_content.append({"type": "text", "text": error_msg_detail})
            metadata["error"] = "Framework Error"
            metadata["error_type"] = type(e).__name__

        if not result_content:
             result_content.append({"type": "text", "text": "Execution finished, but no specific result or error was generated."})

        return {"content": result_content}, reward, metadata
    
    async def release(self, instance_id: str, **kwargs) -> None:
        """Releases resources associated with an instance."""
        log_extra = {'instance_id': instance_id}
        instance_data = self._instance_dict.pop(instance_id, None)
        
        if instance_data:
            worker_id = instance_data.get("worker_id")
            temp_dir = instance_data.get("temp_dir")
            
            if worker_id is not None:
                payload = {'instance_id': instance_id}
                try:
                    await self._send_request(worker_id, 'release', payload, timeout=5.0)
                except Exception as e:
                    logger.warning(f"release() fallback – worker {worker_id} did not acknowledge: {e}", extra=log_extra)
                    try:
                        # To prevent blocking the event loop, run the blocking put() in an executor.
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(None, self.task_queues[worker_id].put, (None, 'release', payload))
                    except Exception as put_e:
                        logger.error(f"Failed to send release command to worker {worker_id} via fallback: {put_e}", extra=log_extra)

            if temp_dir and os.path.exists(temp_dir):
                try: 
                    shutil.rmtree(temp_dir)
                except OSError as e: 
                    logger.error(f"Error removing temp_dir '{temp_dir}': {e}", extra=log_extra)
            
            gc.collect()
        else:
             logger.warning(f"Attempted to release non-existent instance '{instance_id}'.", extra=log_extra)

    async def restart_all_workers(self, **kwargs) -> bool:
        """
        Terminates all worker processes and restarts them. Clears all active instances.
        This is a hard reset for the entire tool, designed to be called between training batches.
        """
        logger.info("Restarting all CodeExecuteTool workers...")

        # 1. Terminate all existing workers
        for i, worker in enumerate(self.workers):
            if worker.is_alive():
                try:
                    self.task_queues[i].put(None)
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, worker.join, 10) # 10s timeout
                    if worker.is_alive():
                        logger.warning(f"Worker {i} (pid {worker.pid}) did not terminate gracefully, killing.")
                        worker.kill()
                        await loop.run_in_executor(None, worker.join)
                except Exception as e:
                    logger.error(f"Error terminating worker {i}: {e}")

        # 2. Clean up internal state
        self._instance_dict.clear()
        
        # Cancel pending futures
        for future in self.pending_futures.values():
            if not future.done():
                future.cancel("Restarting all workers")
        self.pending_futures.clear()

        # Cancel and cleanup the result reader task
        if self._result_reader_task and not self._result_reader_task.done():
            self._result_reader_task.cancel()
            try:
                await self._result_reader_task
            except asyncio.CancelledError:
                pass # Expected
        self._result_reader_task = None
        
        # Cleanup old temp directories for all instances
        if os.path.exists(TEMP_DIR_BASE):
            shutil.rmtree(TEMP_DIR_BASE)
        os.makedirs(TEMP_DIR_BASE, exist_ok=True)
        
        # 3. Re-create queues and workers
        self.workers.clear()
        self.task_queues.clear()
        
        # Create a new result queue to discard any lingering messages from old workers
        self.result_queue = Queue()

        for i in range(self.max_workers):
            task_queue = Queue()
            worker = Process(
                target=worker_main_loop,
                args=(task_queue, self.result_queue),
                name=f"CodeExecWorker-{i}"
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            self.task_queues.append(task_queue)

        logger.info("All CodeExecuteTool workers have been restarted.")
        return True

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 1.0

# NEW: Helper to save image in the main process instead of delegating to the worker.

def _save_image_in_main_process(instance_id: str, image_input: Union[Image.Image, str, dict], temp_dir: str) -> Optional[str]:

    log_extra = {"instance_id": instance_id}
    image_filename = "input_image.jpg"
    try:
    #     if isinstance(image_input, Image.Image):
    #         pil_image = image_input
    #     elif isinstance(image_input, str):
    #         if image_input.startswith("data:image/"):
    #             header, data = image_input.split(",", 1)
    #             pil_image = Image.open(io.BytesIO(base64.b64decode(data)))
    #         elif os.path.exists(image_input):
    #             pil_image = Image.open(image_input)
    #         else:
    #             # Try raw base64 (no data: URI header)
    #             try:
    #                 pil_image = Image.open(io.BytesIO(base64.b64decode(image_input)))
    #             except Exception:
    #                 logger.warning(
    #                     f"Could not interpret provided string as path or base64 for '{instance_id}'.",
    #                     extra=log_extra,
    #                 )
    #                 return None
    #     elif isinstance(image_input, dict) and "bytes" in image_input:
    #         pil_image = Image.open(io.BytesIO(image_input["bytes"]))
    #     else:
    #         logger.warning(
    #             f"Unsupported image input type '{type(image_input)}' for '{instance_id}'.",
    #             extra=log_extra,
    #         )
    #         return None

    #     if pil_image.mode in ("RGBA", "LA", "PA"):
    #         pil_image = pil_image.convert("RGB")
    #     # Handle palletized images with transparency
    #     elif pil_image.mode == "P" and "transparency" in pil_image.info:
    #          pil_image = pil_image.convert("RGB")
    #     # Ensure any other non-RGB mode is converted
    #     elif pil_image.mode != "RGB":
    #          pil_image = pil_image.convert("RGB")
        # make sure the same size 
        pil_image = process_image(image_input)
        os.makedirs(temp_dir, exist_ok=True)
        image_path = os.path.join(temp_dir, image_filename)
        pil_image.save(image_path, format="JPEG")
        logger.info(
            f"[main] Saved input image to '{image_path}' for instance '{instance_id}'.",
        )
        return image_path
    except Exception as e:
        logger.error(
            f"[main] Failed to save image for instance '{instance_id}': {e}",
            exc_info=True,
            extra=log_extra,
        )
        return None

if __name__ == '__main__':
    # --- Mocks ---
    class MockFunction:
        def __init__(self, name: str):
            self.name = name

    class MockOpenAIFunctionToolSchema:
        def __init__(self, name: str, description: str, parameters: dict):
            self.function = MockFunction(name=name)
            self.description = description
            self.parameters = parameters
            self.name = name
        def dict(self):
            return {
                "name": self.function.name,
                "description": self.description,
                "parameters": self.parameters
            }
        def model_dump(self, **kwargs):
            return self.dict()

    class MockBaseTool:
        def __init__(self, config: dict, tool_schema: Optional[MockOpenAIFunctionToolSchema] = None):
            self.config = config
            self.tool_schema = tool_schema
            self.name = tool_schema.function.name if tool_schema else "MockTool"

    BaseTool = MockBaseTool
    OpenAIFunctionToolSchema = MockOpenAIFunctionToolSchema
    # --- End Mocks ---

    async def main_test():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s')
        
        schema = MockOpenAIFunctionToolSchema(
            name="AdvancedPythonCodeExecutor",
            description="Executes Python code with plotting and context support.",
            parameters={
                "type": "object",
                "properties": {
                    "instance_id": {"type": "string"},
                    "code": {"type": "string"},
                    "timeout": {"type": "number", "default": DEFAULT_TIMEOUT_SECONDS}
                }, "required": ["instance_id", "code"]
            }
        )

        # The test must run within a single event loop.
        # CodeExecuteTool now starts its result reader task upon instantiation.
        tool = CodeExecuteTool(config={"max_workers": 1}, tool_schema=schema)

        print(f"Tool Schema used by instance: {tool.tool_schema.dict()}")

        inst1_id = None
        inst2_id = None
        # NOTE: Wrapped demo code below previously used a try/except/finally block.
        # The 'except' and 'finally' sections were removed for brevity in earlier edits,
        # which left a lone 'try:' causing a syntax error. Commenting it out restores
        # syntactic correctness without affecting library functionality.
        # try:
        # --- Instance 1: Plotting and context sharing ---
        inst1_id = await tool.create()
        print(f"\n--- Test Case: Instance 1 ({inst1_id}) ---")

        code1_part1 = """
import matplotlib.pyplot as plt
import math
print('Instance 1, Part 1: Plotting a line graph...')
my_variable = 42
plt.figure(figsize=(4, 2.5))
plt.plot([10, 20, 30], [1, 4, 2], marker='o')
plt.title('Instance 1 - Line Plot')
plt.xlabel('X-axis'); plt.ylabel('Y-axis')
plt.grid(True);
print(f'Line plot generated. my_variable is {my_variable}')
plt.show()
x = 100
x = 5 * math.sqrt(2)
x_rounded = round(x, 1) 
while True:
    pass
"""
        res1_part1, reward1_part1, meta1_part1 = await tool.execute(inst1_id, {"code": code1_part1})
        print(f"Instance 1, Part 1 Result (Reward: {reward1_part1}): {meta1_part1}")
        for item in res1_part1['content']:
            content_preview = item.get('text', item.get('image', ''))
            if item['type'] == 'image':
                content_preview = content_preview[:70] + '...'
            print(f"  Output Type: {item['type']}, Content: {content_preview}")

        code1_part2 = """
import matplotlib.pyplot as plt
print(x)
print('\\nInstance 1, Part 2: Accessing context and plotting...')
if 'my_variable' in locals():
    print(f'Successfully accessed my_variable: {my_variable}')
    new_var = my_variable * 2
else:
    print('Error: my_variable not found in context!')
    new_var = 0

plt.figure(figsize=(4, 2.5))
plt.scatter([1, 2, 3, 4], [new_var/20, new_var/15, new_var/30, new_var/10], color='red')
plt.title('Instance 1 - Scatter Plot')
print(f'Scatter plot generated. new_var is {new_var}')
plt.show()
"""
        res1_part2, reward1_part2, meta1_part2 = await tool.execute(inst1_id, {"code": code1_part2})
        print(f"Instance 1, Part 2 Result (Reward: {reward1_part2}): {meta1_part2}")
        for item in res1_part2['content']:
            content_preview = item.get('text', item.get('image', ''))
            if item['type'] == 'image':
                content_preview = content_preview[:70] + '...'
            print(f"  Output Type: {item['type']}, Content: {content_preview}")


        # --- Instance 2: Image input ---
        from PIL import Image
        dummy_img_data = Image.new('RGB', (1024, 1024), color = 'red')
        inst2_id = await tool.create(image=dummy_img_data)
        print(f"\n--- Test Case: Instance 2 ({inst2_id}) with image input ---")

        code_img_proc = (
            """
from PIL import Image
bbox_2d = [1683, 1138, 1710, 1213]
img = Image.open('input_image.jpg')
sub_img = img.crop(bbox_2d)
sub_img
            """
        )
        res_img, reward_img, meta_img = await tool.execute(inst2_id, {"code": code_img_proc})
        print(f"Instance 2 Image Proc Result (Reward: {reward_img}): {meta_img}")
        for item in res_img['content']:
            content_preview = item.get('text', item.get('image', ''))
            if item['type'] == 'image':
                content_preview = content_preview
            print(f"  Output Type: {item['type']}, Content: {content_preview}")


        code_img_proc1 = ("""from PIL import Image

img = Image.open('input_image.jpg')
bbox_2d = [1640, 688, 1668, 713]
sub_img = img.crop(bbox_2d)

x1, y1, x2, y2 = bbox_2d
bbox_w = x2 - x1
bbox_h = y2 - y1

# Zoom-in image
if bbox_w < 28 or bbox_h < 28:
    if bbox_w > 0 and bbox_h > 0:
        scale = 56 / min(bbox_w, bbox_h)
        new_w = int(bbox_w * scale)
        new_h = int(bbox_h * scale)
        result_img = sub_img.resize((new_w, new_h), Image.LANCZOS)
    else:
        result_img = sub_img
else:
    result_img = sub_img

result_img
        
        """)
        res_img, reward_img, meta_img = await tool.execute(inst2_id, {"code": code_img_proc1})
        print(f"Instance 2 Image Proc Result----------double (Reward: {reward_img}): {meta_img}")

        for item in res_img['content']:
            content_preview = item.get('text', item.get('image', ''))
            if item['type'] == 'image':
                content_preview = content_preview
            print(f"  Output Type: {item['type']}, Content: {content_preview}")

        # except Exception as e:
        #      logging.exception("An error occurred during the test execution.")
        # finally:
        print("\n--- Releasing instances ---")
        if inst1_id:
            await tool.release(inst1_id)
        if inst2_id:
            await tool.release(inst2_id)


    # Using 'spawn' is safer across platforms to avoid issues with forked processes
    # inheriting state from libraries like matplotlib, which can cause deadlocks.
    # We set this for all platforms, not just darwin.
    multiprocessing.set_start_method('spawn', force=True)

    # Run the async main function
    asyncio.run(main_test())