# Merged Executor - Direct results collection without logfire

import os
import json
import logging
from datetime import datetime, timezone
import time
import asyncio
import uuid
import pandas as pd
import logfire
from pydantic import BaseModel
from accelopt.utils import extract_first_code, construct_query_coroutine, retry_query_coroutine
from accelopt.flb_wrapper import FlashInferKernel, format_definition, get_unique_trace_name
from flashinfer_bench import Solution, Definition, Trace
from flashinfer_bench.data import load_json_file, save_json_file
from agents import AsyncOpenAI, set_tracing_disabled
from functools import partial

# -------------------------- Logging --------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------- Config Models --------------------------
class FixerPromptConfig(BaseModel):
    model_name: str = ""
    system_prompt: str = ""
    user_template: str = ""
    log: str = ""
    code: str = ""

class ExecutorPromptConfig(BaseModel):
    model_name: str = ""
    system_prompt: str = ""
    definition_path: str = ""
    workload_path: str = ""
    baseline_solution_path: str = ""
    user_template: str = ""
    optimization_plan: str = ""
    save_fields: list[str] = []

class ExecutorConfig(BaseModel):
    model_name: str = ""
    system_prompt: str = ""
    service_name: str = ""
    definition_path: str = ""
    workload_path: str = ""
    baseline_solution_path: str = ""
    baseline_trace_path: str = ""
    optimization_plan: str = ""
    num_samples: int = 4
    user_template: str = ""
    save_fields: list[str] = []
    traceset_root: str = ""
    fixer_steps: int = 0
    fixer_system_prompt: str = ""
    fixer_user_template: str = ""

# -------------------------- Helpers --------------------------
def construct_executor_prompt(config: ExecutorPromptConfig) -> str:
    definition = load_json_file(Definition, config.definition_path)
    solution = load_json_file(Solution, config.baseline_solution_path)
    user_prompt = (
        config.user_template
        .replace("{problem_code}", format_definition(definition))
        .replace("{kernel_code}", solution.sources[0].content)
        .replace("{optimization_plan}", config.optimization_plan)
    )
    return user_prompt

def construct_fixer_prompt(log, code, fixer_user_template: str) -> str:
    user_prompt = (
        fixer_user_template
        .replace("{log}", log)
        .replace("{code}", code)
    )
    return user_prompt

# -------------------------- Parallel LLM --------------------------
async def propose_once(name: str, config: ExecutorPromptConfig, client: AsyncOpenAI):
    try:
        user_prompt = construct_executor_prompt(config)
        kwargs = {}
        if "gpt-oss" in config.model_name.lower():
            kwargs['extra_body'] = {'reasoning_effort': 'medium'}
            kwargs['max_tokens'] = 65536
        elif "claude" in config.model_name.lower():
            kwargs['temperature'] = 1.0  # Temperature must be 1.0 for reasoning to be enabled
            kwargs['max_tokens'] = 20000
            kwargs['extra_body'] = {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 10000
                }
            }
        elif "gemini" in config.model_name.lower():
            kwargs['max_tokens'] = 65536
            kwargs['extra_body'] = {
                'extra_body':{
                    'google': {
                        'thinking_config': {
                            'thinking_level': 'high'
                        }
                    }
                }
            }
        else:
            kwargs = {}
        q_co = partial(construct_query_coroutine, client, config.model_name, config.system_prompt, user_prompt, **kwargs)
        response = await retry_query_coroutine(q_co, max_retries=15, delay=10)
        if response is None:
            return None
        code = extract_first_code(response.choices[0].message.content, ["python"])
        if not code:
            return None
        return {"name": name, "response": response, "code": code}
    except asyncio.TimeoutError:
        logger.warning("LLM timed out for %s", name)
        return None
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("propose_once failed for %s", name)
        return None

async def fix_once(name, config: FixerPromptConfig, client: AsyncOpenAI):
    try:
        user_prompt = construct_fixer_prompt(config.log, config.code, config.user_template)
        kwargs = {}
        if "gpt-oss" in config.model_name.lower():
            kwargs['extra_body'] = {'reasoning_effort': 'medium'}
            kwargs['max_tokens'] = 65536
        elif "claude" in config.model_name.lower():
            kwargs['temperature'] = 1.0  # Temperature must be 1.0 for reasoning to be enabled
            kwargs['max_tokens'] = 20000
            kwargs['extra_body'] = {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 10000
                }
            }
        elif "gemini" in config.model_name.lower():
            kwargs['max_tokens'] = 65536
            kwargs['extra_body'] = {
                'extra_body':{
                    'google': {
                        'thinking_config': {
                            'thinking_level': 'medium'
                        }
                    }
                }
            }
        else:
            kwargs = {}
        q_co = partial(construct_query_coroutine, client, config.model_name, config.system_prompt, user_prompt, **kwargs)
        response = await retry_query_coroutine(q_co, max_retries=15, delay=10)
        if response is None:
            return None
        code = extract_first_code(response.choices[0].message.content, ["python"])
        if not code:
            return None
        return code
    except asyncio.TimeoutError:
        logger.warning("LLM timed out for %s", name)
        return None
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("fix_once failed for %s", name)
        return None

# ---------- Stage 1: parallel LLM (async) ----------
async def stage1_gather_proposals(service_name: str, pconfig: ExecutorPromptConfig, client: AsyncOpenAI, num_samples: int):
    tasks = []
    for i in range(num_samples):
        tasks.append(asyncio.create_task(propose_once(f"{service_name}_{i}", pconfig, client)))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if isinstance(r, dict)]

async def stage3_fix_code(name, config: FixerPromptConfig, client: AsyncOpenAI):
    code = await fix_once(name, config, client)
    return code

# ---------- Stage 2: sequential profiling with result collection ----------
async def stage2_profile_and_collect(
    proposals: list[dict],
    case_config: ExecutorConfig,
    fixer_config: FixerPromptConfig,
    fixer_client: AsyncOpenAI,
    per_profile_timeout: int = 900
):
    results = []
    num_fixed_iters = case_config.fixer_steps
    
    bl = load_json_file(Trace, case_config.baseline_trace_path).evaluation.performance.latency_ms
    for prop_id, prop in enumerate(proposals):
        name = prop["name"]
        code = prop["code"]
        fix_iter = 0
        while True:
            new_solution_path = None
            profile_trace_path = None
            try:
                start = time.monotonic()
                fixer_name = f"{name}_Fixer_{fix_iter}"
                print(f"[Stage2] START name={fixer_name} timeout={per_profile_timeout}s")
                kernel = FlashInferKernel(
                    traceset_root=case_config.traceset_root,
                    definition_path=case_config.definition_path
                )
                
                new_solution_path = kernel.create_and_save_solution(
                    code=code,
                    author="AccelOpt",
                    language="triton",
                    target_gpu="H100",
                    name=uuid.uuid4().hex, # Filename has to have fewer than 255 characters
                    description=f"{case_config.service_name}_{prop_id}"
                )
                profile_trace, kp = kernel.profile(
                    solution_path=new_solution_path,
                    workload_path=case_config.workload_path,
                    timeout_seconds=per_profile_timeout,
                    profile_baseline=False,
                    use_isolated_runner=True
                )
                profile_trace_path = os.path.join(
                    case_config.traceset_root,
                    "traces",
                    kernel.definition.op_type,
                    get_unique_trace_name(
                        load_json_file(Solution, new_solution_path),
                        load_json_file(Trace, case_config.workload_path),
                    ) + ".json"
                )
                save_json_file(profile_trace, profile_trace_path)
                record_result = {
                    "definition_path": case_config.definition_path,
                    "workload_path": case_config.workload_path,
                    "baseline_solution_path": case_config.baseline_solution_path,
                    "baseline_trace_path": case_config.baseline_trace_path,
                    "solution_path": new_solution_path,
                    "trace_path": profile_trace_path,
                    "baseline_latency": bl,
                }

                # Check if there were errors
                # if not kp.get("compiled", False) or not kp.get("runnable", False) or not kp.get("correct", False):
                
                if not kp.compiled or not kp.runnable or not kp.correct:
                    metadata = kp.metadata
                    error_msg = (metadata.get("compilation_error") or 
                            metadata.get("correctness_error") or 
                            metadata.get("run_error") or 
                            "Unknown error")
                    record_result["error"] = error_msg
                    if fix_iter < num_fixed_iters:
                        fixer_config.log = profile_trace.evaluation.log
                        fixer_config.code = code
                        code = await stage3_fix_code(fixer_name, fixer_config, fixer_client)
                        fix_iter += 1
                        elapsed = time.monotonic() - start
                        print(f"[Stage2] END name={fixer_name} elapsed={elapsed}s")
                        if code is not None:
                            continue
                    else:
                        break
                else:
                    # Success case - add latency and speedup
                    metadata = kp.metadata
                    record_result["latency"] = metadata.get("latency")
                    
                    
                    cl = metadata.get("latency")
                    if bl and cl:
                        record_result["speedup"] = bl / cl
                    else:
                        record_result["speedup"] = None
                    elapsed = time.monotonic() - start
                    print(f"[Stage2] END name={fixer_name} elapsed={elapsed}s")
                    break
                
            except Exception as e:
                logger.error(f"[Profile Error] {fixer_name}: {e}")
                record_result = {
                    "definition_path": case_config.definition_path,
                    "workload_path": case_config.workload_path,
                    "baseline_solution_path": case_config.baseline_solution_path,
                    "baseline_trace_path": case_config.baseline_trace_path,
                    "solution_path": new_solution_path,
                    "trace_path": profile_trace_path,
                    "error": str(e),
                    "baseline_latency": bl
                }
                break
        results.append(record_result)
    
    return results

# ---------- main(): orchestrates proposal generation and profiling ----------
async def process_single_service_plan(
    case_config: ExecutorConfig, 
    fixer_config: FixerPromptConfig,
    executor_client: AsyncOpenAI,
    fixer_client: AsyncOpenAI
):
    pconfig = ExecutorPromptConfig(
        model_name=case_config.model_name,
        system_prompt=case_config.system_prompt,
        definition_path=case_config.definition_path,
        workload_path=case_config.workload_path,
        baseline_solution_path=case_config.baseline_solution_path,
        user_template=case_config.user_template,
        optimization_plan=case_config.optimization_plan,
        save_fields=case_config.save_fields,
    )

    # 1) LLM parallel proposal generation
    proposals = await stage1_gather_proposals(case_config.service_name, pconfig, executor_client, case_config.num_samples)
    
    # 2) Sequential profiling with result collection
    results = await stage2_profile_and_collect(proposals, case_config, fixer_config, fixer_client, per_profile_timeout=180)
    
    return results

# -------------------------- Driver --------------------------
async def main(args):
    # time record (start)
    os.makedirs(args.exp_dir, exist_ok=True)
    start_time = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')

    # load inputs
    with open(args.base_prompt_path, "r") as f:
        system_prompt = f.read()
    with open(args.user_template_path, "r") as f:
        user_template = f.read()
    with open(args.problems_path, "r") as f:
        problems = pd.read_csv(f)
    with open(args.extractor_output_path, "r") as f:
        extractor_output_list = json.load(f)
    if args.save_fields_path is not None:
        with open(args.save_fields_path, "r") as f:
            save_fields = json.load(f)
    else:
        save_fields = []
    with open(args.model_config_path, "r") as f:
        model_config = json.load(f)

    with open(args.fixer_base_prompt_path, "r") as f:
        fixer_system_prompt = f.read()
    with open(args.fixer_user_template_path, "r") as f:
        fixer_user_template = f.read()

    # model client
    LLM_TIMEOUT = 60000
    BASE_URL = model_config['executor']['url']
    API_KEY = model_config['executor']['api_key']
    executor_client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=LLM_TIMEOUT)

    BASE_URL = model_config['fixer']['url']
    API_KEY = model_config['fixer']['api_key']
    fixer_client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=LLM_TIMEOUT)

    # Collect all results
    output_list = []

    # iterate services
    for _, row in problems.iterrows():
        baseline_solution_path = row["solution_path"]
        plans = next(item["plans"] for item in extractor_output_list if item["baseline_solution_path"] == baseline_solution_path)
        # profile baseline once per service (blocking)

        single_dict = {"baseline_solution_path": baseline_solution_path, "workload_path": row["workload_path"]}
        logfire_service_name = load_json_file(Solution, baseline_solution_path).name
        for i in range(len(plans)):
            plan = plans[i]
            case_config = ExecutorConfig(
                system_prompt=system_prompt,
                model_name=model_config['executor']['model'],
                service_name=f"{logfire_service_name}_plan_{i}",
                optimization_plan=plan,
                definition_path=row["definition_path"],
                workload_path=row["workload_path"],
                baseline_solution_path=baseline_solution_path,
                baseline_trace_path=row["trace_path"],
                num_samples=args.num_samples,
                user_template=user_template,
                save_fields=save_fields,
                traceset_root=args.traceset_root,
                fixer_steps=args.fixer_steps,
                fixer_system_prompt=fixer_system_prompt,
                fixer_user_template=fixer_user_template
            )

            fixer_config = FixerPromptConfig(
                model_name=model_config['fixer']['model'],
                system_prompt=fixer_system_prompt,
                user_template=fixer_user_template,
                log="",
                code=""
            )
            
            plan_results = await asyncio.wait_for(
                process_single_service_plan(case_config, fixer_config, executor_client, fixer_client), 
                timeout=7200
            )
            
            # Store results for each sample in this plan
            for j in range(args.num_samples):
                if j < len(plan_results):
                    single_dict[f"plan_{i}_{j}"] = plan_results[j]
                else:
                    single_dict[f"plan_{i}_{j}"] = {"error": "No implementation found"}

        output_list.append(single_dict)

    # time record (end)
    end_time = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')

    # Save results in format similar to read_executor_log output
    output_dict = {
        "exp_date": args.exp_date,
        "executor_dir": args.exp_dir,
        "executor_start_timestamp": start_time,
        "executor_end_timestamp": end_time,
        "executor_results": output_list
    }

    with open(args.output_path, "w") as f:
        json.dump(output_dict, f, indent=4)

    # Create combined timing file for compatibility
    combined_time_record_path = f"{args.exp_dir}/executor_start_end_time.txt"
    with open(combined_time_record_path, "w") as f:
        f.write(f"{start_time},{end_time}")

    logger.info(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--problems_path", type=str, required=True)
    parser.add_argument("--extractor_output_path", type=str, required=True)
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--base_prompt_path", type=str, required=True)
    parser.add_argument("--user_template_path", type=str, required=True)
    parser.add_argument("--save_fields_path", type=str, required=False, default=None)
    parser.add_argument("--model_config_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--exp_date", type=str, required=True)
    parser.add_argument("--traceset_root", type=str, required=True)
    parser.add_argument("--fixer_steps", type=int, required=False, default=0)
    parser.add_argument("--fixer_base_prompt_path", type=str, required=True)
    parser.add_argument("--fixer_user_template_path", type=str, required=True)
    args = parser.parse_args()


    set_tracing_disabled(disabled=True)
    logfire.configure()
    logfire.instrument_openai()
    asyncio.run(main(args))