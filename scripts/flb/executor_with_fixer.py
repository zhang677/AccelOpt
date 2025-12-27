# Merged Executor - Direct results collection without logfire

import os
import json
import logging
from datetime import datetime, timezone
import multiprocessing as mp
import time
import asyncio
import uuid
import pandas as pd
from pydantic import BaseModel
import logfire
from accelopt.utils import extract_first_code, retry_runner_safer
from accelopt.flb_wrapper import FlashInferKernel, format_definition, get_unique_trace_name
from flashinfer_bench import Solution, Definition, Trace
from flashinfer_bench.data import load_json_file, save_json_file
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, RunConfig, ModelSettings

# -------------------------- Logging --------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------- Config Models --------------------------
class ExecutorPromptConfig(BaseModel):
    definition_path: str = ""
    workload_path: str = ""
    baseline_solution_path: str = ""
    user_template_path: str = ""
    optimization_plan: str = ""
    save_fields: list[str] = []

class ExecutorConfig(BaseModel):
    system_prompt: str = ""
    service_name: str = ""
    definition_path: str = ""
    workload_path: str = ""
    baseline_solution_path: str = ""
    baseline_trace_path: str = ""
    optimization_plan: str = ""
    num_samples: int = 4
    user_template_path: str = ""
    save_fields: list[str] = []
    traceset_root: str = ""
# -------------------------- Helpers --------------------------
def construct_executor_prompt(config: ExecutorPromptConfig) -> str:
    definition = load_json_file(Definition, config.definition_path)
    solution = load_json_file(Solution, config.baseline_solution_path)
    with open(config.user_template_path, "r") as f:
        prompt_template = f.read()
    user_prompt = (
        prompt_template
        .replace("{problem_code}", format_definition(definition))
        .replace("{kernel_code}", solution.sources[0].content)
        .replace("{optimization_plan}", config.optimization_plan)
    )
    return user_prompt

def construct_fixer_prompt(log, code) -> str:
    user_prompt = f"""
Wrong code:
```
{code}
```

Error log:
{log}

Output the fixed `run` function and all the functions it calls wrapped in code block.
"""
    return user_prompt

# -------------------------- Parallel LLM --------------------------
async def propose_once(name: str, config: ExecutorPromptConfig, agent: Agent):
    try:
        user_prompt = construct_executor_prompt(config)
        if "claude" in agent.model.model.lower():
            run_config = RunConfig(
                model_settings=ModelSettings(
                    temperature=1.0, # Temperature must be 1.0 for reasoning to be enabled
                    max_tokens=20000,
                    extra_body={
                        "thinking": {
                            "type": "enabled",
                            "budget_tokens": 10000
                        }
                    }
                )
            )
        else:
            run_config = None
        result = await retry_runner_safer(agent, user_prompt, run_config=run_config, max_retries=15, delay=10)
        if result is None:
            return None
        code = extract_first_code(result.final_output, ["python"])
        if not code:
            return None
        return {"name": name, "result": result, "code": code}
    except asyncio.TimeoutError:
        logger.warning("LLM timed out for %s", name)
        return None
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("propose_once failed for %s", name)
        return None

async def fix_once(name, log, code, agent: Agent):
    try:
        user_prompt = construct_fixer_prompt(log, code)
        if "claude" in agent.model.model.lower():
            run_config = RunConfig(
                model_settings=ModelSettings(
                    temperature=1.0, # Temperature must be 1.0 for reasoning to be enabled
                    max_tokens=20000,
                    extra_body={
                        "thinking": {
                            "type": "enabled",
                            "budget_tokens": 10000
                        }
                    }
                )
            )
        else:
            run_config = None
        result = await retry_runner_safer(agent, user_prompt, run_config=run_config, max_retries=15, delay=10)
        if result is None:
            return None
        code = extract_first_code(result.final_output, ["python"])
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
async def stage1_gather_proposals(service_name: str, pconfig: ExecutorPromptConfig, base_agent: Agent, num_samples: int):
    tasks = []
    for i in range(num_samples):
        # fresh agent per task to avoid cross-cancellation
        agent = Agent(name=f"Executor_{i}", instructions=base_agent.instructions, model=base_agent.model)
        tasks.append(asyncio.create_task(propose_once(f"{service_name}_{i}", pconfig, agent)))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if isinstance(r, dict)]

async def stage3_fix_code(name, log, error_code, model: OpenAIChatCompletionsModel):
    fixer_system_prompt = """
You are a helpful assistant that fixes the code given the error log.

Here is some information about Triton:
1. triton.language.dot(input, other, acc=None):
Returns the matrix product of two blocks. The two blocks must both be two-dimensional or three-dimensional and have compatible inner dimensions. For three-dimensional blocks, tl.dot performs the batched matrix product, where the first dimension of each block represents the batch dimension.

Parameters:
- input (2D or 3D tensor of scalar-type in {int8, float8_e5m2, float16, bfloat16, float32}) – The first tensor to be multiplied.
- other (2D or 3D tensor of scalar-type in {int8, float8_e5m2, float16, bfloat16, float32}) – The second tensor to be multiplied.
- acc (2D or 3D tensor of scalar-type in {float16, float32, int32}) – The accumulator tensor. If not None, the result is added to this tensor.

2. A parameter should be defined either inside the @triton.autotune configuration or as a keyword argument of the kernel. Make sure that you don't re-define auto-tuned symbols.
```
@triton.autotune(configs=[
    triton.Config(kwargs={'BLOCK_SIZE': 128}, num_warps=4),
    triton.Config(kwargs={'BLOCK_SIZE': 1024}, num_warps=8),
  ],
  key=['x_size'] # the two above configs will be evaluated anytime the value of x_size changes
)
@triton.jit
def kernel(x_ptr, x_size, BLOCK_SIZE: tl.constexpr):
  ...
# When you call the kernel, don't pass BLOCK_SIZE as an argument
kernel[grid](x_ptr, x_size)
```

3. Triton doesn't support slice indexing on local accumulators. Therefore, you could use tl.where to nullify or select specific elements and perform operations on the entire tensor.
Example: If you want to zero out the second half of a row:
```
# Assume x has shape [BLOCK_SIZE]
cols = tl.arange(0, BLOCK_SIZE)
mask = cols < (BLOCK_SIZE // 2)

# Simulate slicing x[:BLOCK_SIZE//2] by zeroing the rest
sliced_x = tl.where(mask, x, 0.0)
```

4. The start and end must be a power of two in triton.language.arange(start, end)

5. There are no explicit shared memory management Triton APIs. The compiler manages shared memory. 
"""
    agent = Agent(name=name, instructions=fixer_system_prompt, model=model)
    code = await fix_once(name, log, error_code, agent)
    return code

# ---------- Stage 2: sequential profiling with result collection ----------
async def stage2_profile_and_collect(
    proposals: list[dict],
    case_config: ExecutorConfig,
    model: OpenAIChatCompletionsModel,
    per_profile_timeout: int = 900
):
    results = []
    num_fixed_iters = 4
    
    bl = load_json_file(Trace, case_config.baseline_trace_path).evaluation.performance.latency_ms
    for prop_id, prop in enumerate(proposals):
        name = prop["name"]
        code = prop["code"]
        fix_iter = 0
        while True:
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
                        
                        code = await stage3_fix_code(fixer_name, profile_trace.evaluation.log, code, model)
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
    model: OpenAIChatCompletionsModel
):
    pconfig = ExecutorPromptConfig(
        definition_path=case_config.definition_path,
        workload_path=case_config.workload_path,
        baseline_solution_path=case_config.baseline_solution_path,
        user_template_path=case_config.user_template_path,
        optimization_plan=case_config.optimization_plan,
        save_fields=case_config.save_fields,
    )
    agent = Agent(name="Executor", instructions=case_config.system_prompt, model=model)

    # 1) LLM parallel proposal generation
    proposals = await stage1_gather_proposals(case_config.service_name, pconfig, agent, case_config.num_samples)
    
    # 2) Sequential profiling with result collection
    results = await stage2_profile_and_collect(proposals, case_config, model, per_profile_timeout=180)
    
    return results

# -------------------------- Driver --------------------------
async def main(args):
    # time record (start)
    os.makedirs(args.exp_dir, exist_ok=True)
    start_time = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')

    # load inputs
    with open(args.base_prompt_path, "r") as f:
        system_prompt = f.read()
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

    # model client
    BASE_URL = model_config['url']
    API_KEY = model_config['api_key']
    LLM_TIMEOUT = 60000
    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=LLM_TIMEOUT)
    model = OpenAIChatCompletionsModel(model=model_config["model"], openai_client=client)
    set_tracing_disabled(disabled=True)

    # pin visible core
    # mp.set_start_method("spawn", force=True)

    # Collect all results
    output_list = []

    # iterate services
    for _, row in problems.iterrows():
        baseline_solution_path = row["solution_path"]
        logfire_service_name = load_json_file(Solution, baseline_solution_path).name
        logfire.configure(service_name=logfire_service_name)
        logfire.instrument_openai()
        plans = next(item["plans"] for item in extractor_output_list if item["baseline_solution_path"] == baseline_solution_path)
        # profile baseline once per service (blocking)

        single_dict = {"baseline_solution_path": baseline_solution_path, "workload_path": row["workload_path"]}
        logfire_service_name = load_json_file(Solution, baseline_solution_path).name
        for i in range(len(plans)):
            plan = plans[i]
            case_config = ExecutorConfig(
                system_prompt=system_prompt,
                service_name=f"{logfire_service_name}_plan_{i}",
                optimization_plan=plan,
                definition_path=row["definition_path"],
                workload_path=row["workload_path"],
                baseline_solution_path=baseline_solution_path,
                baseline_trace_path=row["trace_path"],
                num_samples=args.num_samples,
                user_template_path=args.user_template_path,
                save_fields=save_fields,
                traceset_root=args.traceset_root
            )
            
            plan_results = await asyncio.wait_for(
                process_single_service_plan(case_config, model), 
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
    parser.add_argument("--num_samples", type=int, default=4)
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
    args = parser.parse_args()

    asyncio.run(main(args))