import re
import uuid, asyncio
from agents import Runner, ModelBehaviorError
from pydantic import ValidationError

def get_case_name(problem_name, values):
    sorted_keys = sorted(values.keys())
    case_name = problem_name + "_" + "_".join([f"{k}{values[k]}" for k in sorted_keys])
    return case_name

def init_service_name(case_name: str):
    assert "_ID" not in case_name, f"{case_name} already has ID"
    return case_name + f"_ID{uuid.uuid4().hex}"


async def retry_runner_safer(agent, prompt, run_config=None, max_retries=3, delay=3):
    for attempt in range(1, max_retries + 1):
        t = asyncio.create_task(Runner.run(agent, prompt, run_config=run_config))
        try:
            return await t

        except asyncio.CancelledError:
            if t.cancelled():
                try:
                    await asyncio.sleep(delay)  # if WE are cancelled here, it will propagate
                except asyncio.CancelledError:
                    raise
                continue
            # Our task was cancelled by a parent (wait_for/shutdown) → propagate.
            t.cancel(); raise
        except (ModelBehaviorError, ValidationError):
            if attempt == max_retries:
                return None
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                raise
            continue
    return None

def extract_first_code(output_string: str, code_language_types: list[str]) -> str | None:
    """
    Extract first code block from model output, specified by code_language_type
    """
    trimmed = output_string.strip()

    # Extracting the first occurrence of content between backticks
    code_match = re.search(r"```(.*?)```", trimmed, re.DOTALL)

    if code_match:
        # Strip leading and trailing whitespace from the extracted code
        code = code_match.group(1).strip()

        # depends on code_language_type: cpp, python, etc.
        # sometimes the block of code is ```cpp ... ``` instead of ``` ... ```
        # in this case strip the cpp out
        for code_type in code_language_types:
            if code.startswith(code_type):
                code = code[len(code_type) :].strip()

        return code

    return None