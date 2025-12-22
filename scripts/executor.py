# Merged Executor - Direct results collection without logfire

import os
import json
import tempfile
import traceback
import contextlib
import logging
from datetime import datetime, timezone
import multiprocessing as mp
import time
import asyncio
import pandas as pd
from pydantic import BaseModel
import logfire
from accelopt.utils import extract_first_code, retry_runner_safer
from accelopt.kernel_wrapper import NKIKernel
from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, RunConfig, ModelSettings

# -------------------------- Logging --------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------- Config Models --------------------------
class ExecutorPromptConfig(BaseModel):
    host_numpy_path: str = ""
    nki_kernel_path: str = ""
    user_template_path: str = ""
    optimization_plan: str = ""
    save_fields: list[str] = []

class ExecutorConfig(BaseModel):
    system_prompt: str = ""
    service_name: str = ""
    kernel_path: str = ""
    task_path: str = ""
    optimization_plan: str = ""
    problem: str = ""
    values: str = ""
    case_name: str = ""
    num_samples: int = 4
    user_template_path: str = ""
    save_fields: list[str] = []
    rel_tol: float = 2e-5
# -------------------------- Helpers --------------------------
def construct_executor_prompt(config: ExecutorPromptConfig) -> str:
    with open(config.host_numpy_path, "r") as f:
        host_numpy_function = f.read()
    with open(config.nki_kernel_path, "r") as f:
        nki_kernel_function = f.read()
    with open(config.user_template_path, "r") as f:
        prompt_template = f.read()
    user_prompt = (
        prompt_template
        .replace("{problem_code}", host_numpy_function)
        .replace("{kernel_code}", nki_kernel_function)
        .replace("{optimization_plan}", config.optimization_plan)
    )
    return user_prompt

def _write_temp_kernel(code: str) -> str:
    fd, temp_path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(
            "import numpy as np\n"
            "import neuronxcc.nki as nki\n"
            "import neuronxcc.nki.language as nl\n"
            "import neuronxcc.nki.typing as nt\n"
            "import neuronxcc.nki.isa as nisa\n"
            "from neuronxcc.nki import trace\n"
            "from neuronxcc.nki.language import par_dim\n\n"
            f"{code}\n"
        )
    return temp_path

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

# ---------- Stage 1: parallel LLM (async) ----------
async def stage1_gather_proposals(service_name: str, pconfig: ExecutorPromptConfig, base_agent: Agent, num_samples: int):
    tasks = []
    for i in range(num_samples):
        # fresh agent per task to avoid cross-cancellation
        agent = Agent(name=f"Executor_{i}", instructions=base_agent.instructions, model=base_agent.model)
        tasks.append(asyncio.create_task(propose_once(f"{service_name}_{i}", pconfig, agent)))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if isinstance(r, dict)]

def _profile_worker(program_path: str, base_numpy_path: str, save_fields: list[str], result_path: str, rel_tol: float):
    try:
        k = NKIKernel(program_path, base_numpy_path)
        k.rel_tol = rel_tol
        k.profile(save_fields)
        out = {"compiled": k.res.compiled, "runnable": k.res.runnable, "correct": k.res.correct, "metadata": k.res.metadata or {}}
    except Exception as e:
        out = {"compiled": False, "runnable": False, "correct": False,
               "metadata": {"compilation_error": str(e), "compilation_traceback": traceback.format_exc()}}
    with open(result_path, "w") as f:
        json.dump(out, f)

def profile_with_hard_timeout_sync(program_path: str, base_numpy_path: str, save_fields: list[str], rel_tol: float, timeout_sec: int) -> dict:
    fd, result_path = tempfile.mkstemp(prefix="host_nki_profile_", suffix=".json"); os.close(fd)
    p = mp.Process(target=_profile_worker, args=(program_path, base_numpy_path, save_fields, result_path, rel_tol), daemon=True)
    p.start(); p.join(timeout_sec)
    try:
        if p.is_alive():
            p.terminate(); p.join(5)
            return {"compiled": False, "runnable": False, "correct": False,
                    "metadata": {"compilation_error": f"Hard timeout after {timeout_sec}s"}} 
        with open(result_path) as f:
            return json.load(f)
    except Exception:
        return {"compiled": False, "runnable": False, "correct": False,
                "metadata": {"compilation_error": traceback.format_exc()}}
    finally:
        with contextlib.suppress(Exception): os.remove(result_path)

# ---------- Stage 2: sequential profiling with result collection ----------
def stage2_profile_and_collect(
    proposals: list[dict],
    baseline_kernel: NKIKernel,
    case_config: ExecutorConfig,
    base_spec: dict,
    per_profile_timeout: int = 900
):
    results = []
    for prop in proposals:
        name, result, code = prop["name"], prop["result"], prop["code"]

        spec = {
            "problem": base_spec["problem"],
            "values": base_spec["values"],
            "case_name": base_spec["case_name"],
            "spec_code": base_spec["spec_code"],
            "baseline_code": base_spec["baseline_code"],
            "plan": case_config.optimization_plan,
            "new_kernel_code": code,
            "baseline_latency": baseline_kernel.res.metadata.get("latency"),
            "baseline_metadata": json.dumps(baseline_kernel.res.metadata or {}),
        }

        temp_path = None
        try:
            start = time.monotonic()
            print(f"[Stage2] START name={name} case={base_spec['case_name']} timeout={per_profile_timeout}s")
            temp_path = _write_temp_kernel(code)
            kp = profile_with_hard_timeout_sync(
                program_path=temp_path,
                base_numpy_path=baseline_kernel.base_numpy_path,
                save_fields=case_config.save_fields,
                rel_tol=case_config.rel_tol,
                timeout_sec=per_profile_timeout,
            )
            
            record_result = {
                "body": code,
                "spec_code": spec["spec_code"],
                "baseline": spec["baseline_code"],
                "baseline_latency": spec["baseline_latency"],
                "problem": spec["problem"],
                "values": spec["values"],
                "kernel_metadata": json.dumps(kp.get("metadata", {})),
                "baseline_metadata": spec["baseline_metadata"],
            }

            # Check if there were errors
            if not kp.get("compiled", False) or not kp.get("runnable", False) or not kp.get("correct", False):
                metadata = kp.get("metadata", {})
                error_msg = (metadata.get("compilation_error") or 
                           metadata.get("correctness_error") or 
                           metadata.get("run_error") or
                           "Unknown error")
                record_result["error"] = error_msg
            else:
                # Success case - add latency and speedup
                metadata = kp.get("metadata", {})
                record_result["latency"] = metadata.get("latency")
                
                bl = baseline_kernel.res.metadata.get("latency")
                cl = metadata.get("latency")
                if bl and cl:
                    record_result["speedup"] = bl / cl
                else:
                    record_result["speedup"] = None
            elapsed = time.monotonic() - start
            print(f"[Stage2] END name={name} case={base_spec['case_name']} elapsed={elapsed}s")
            results.append(record_result)
            if record_result.get("error", None) and "Hard timeout" in record_result["error"]:
                print(f"[Stage2] BREAK name={name} case={base_spec['case_name']} elapsed={elapsed}s")
                break
        except Exception as e:
            logger.error(f"[Profile Error] {name}: {e}")
            error_result = {
                "error": str(e),
                "body": code,
                "spec_code": spec["spec_code"],
                "baseline": spec["baseline_code"],
                "baseline_latency": spec["baseline_latency"],
                "problem": spec["problem"],
                "values": spec["values"]
            }
            results.append(error_result)
        finally:
            if temp_path:
                with contextlib.suppress(Exception):
                    os.remove(temp_path)
    
    return results

# ---------- main(): orchestrates proposal generation and profiling ----------
async def process_single_service_plan(
    case_config: ExecutorConfig, 
    baseline_kernel: NKIKernel, 
    model: OpenAIChatCompletionsModel,
    base_spec: dict
):
    pconfig = ExecutorPromptConfig(
        host_numpy_path=case_config.task_path,
        nki_kernel_path=case_config.kernel_path,
        user_template_path=case_config.user_template_path,
        optimization_plan=case_config.optimization_plan,
        save_fields=case_config.save_fields,
    )
    agent = Agent(name="Executor", instructions=case_config.system_prompt, model=model)

    # 1) LLM parallel proposal generation
    proposals = await stage1_gather_proposals(case_config.service_name, pconfig, agent, case_config.num_samples)
    
    # 2) Sequential profiling with result collection
    results = stage2_profile_and_collect(proposals, baseline_kernel, case_config, base_spec, per_profile_timeout=180)
    
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
    with open(args.save_fields_path, "r") as f:
        save_fields = json.load(f)
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
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(args.nc_id)
    mp.set_start_method("spawn", force=True)

    # Collect all results
    output_list = []

    # iterate services
    for _, row in problems.iterrows():
        service_name = row["service_name"]
        logfire.configure(service_name=service_name)
        logfire.instrument_openai()
        plans = next(item["plans"] for item in extractor_output_list if item["service_name"] == service_name)

        # profile baseline once per service (blocking)
        baseline_kernel = NKIKernel(row["kernel"], row["task"])
        baseline_kernel.rel_tol = args.rel_tol
        # baseline_kernel.profile(save_fields)
        baseline_kernel.res.metadata = json.loads(row["profile"])

        with open(row["task"], "r") as f:
            spec_code = f.read()
        with open(baseline_kernel.program_path, "r") as f:
            baseline_code = f.read()

        base_spec = {
            "problem": row["problem"],
            "values": row["values"],
            "case_name": row["case_name"],
            "spec_code": spec_code,
            "baseline_code": baseline_code,
        }

        single_dict = {"service_name": service_name, "case_name": row["case_name"]}
        
        for i in range(len(plans)):
            plan = plans[i]
            case_config = ExecutorConfig(
                system_prompt=system_prompt,
                service_name=f"{service_name}_plan_{i}",
                kernel_path=row["kernel"],
                task_path=row["task"],
                optimization_plan=plan,
                problem=row["problem"],
                values=row["values"],
                case_name=row["case_name"],
                num_samples=args.num_samples,
                user_template_path=args.user_template_path,
                save_fields=save_fields,
                rel_tol=args.rel_tol,
            )
            
            plan_results = await asyncio.wait_for(
                process_single_service_plan(case_config, baseline_kernel, model, base_spec), 
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
        "read_token": args.read_token,
        "executor_start_timestamp": start_time,
        "executor_end_timestamp": end_time,
        "executor_results": output_list
    }

    with open(args.output_path, "w") as f:
        json.dump(output_dict, f, indent=4)

    # Also save timing info (individual and combined for compatibility)
    time_record_path = f"{args.exp_dir}/executor_start_end_time_{args.nc_id}.txt"
    with open(time_record_path, "w") as f:
        f.write(f"{start_time},{end_time}")
    
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
    parser.add_argument("--nc_id", type=int, default=0)
    parser.add_argument("--extractor_output_path", type=str, required=True)
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--base_prompt_path", type=str, required=True)
    parser.add_argument("--user_template_path", type=str, required=True)
    parser.add_argument("--save_fields_path", type=str, required=True)
    parser.add_argument("--model_config_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--exp_date", type=str, required=True)
    parser.add_argument("--read_token", type=str, default="unused")  # For compatibility
    parser.add_argument("--rel_tol", type=float, default=2e-5)
    args = parser.parse_args()

    asyncio.run(main(args))