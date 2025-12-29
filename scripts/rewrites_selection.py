import argparse
from accelopt.utils import retry_runner_safer
import json
import os
import asyncio
from agents import AsyncOpenAI, Agent, OpenAIChatCompletionsModel, set_tracing_disabled, RunConfig, ModelSettings
from pydantic import BaseModel
import logfire
import traceback
import re
import aiohttp

class SummarizerPromptConfig(BaseModel):
    user_template_path: str
    slow_kernel: str
    fast_kernel: str

def construct_summarizer_prompt(config: SummarizerPromptConfig):
    with open(config.user_template_path, "r") as f:
        prompt_template = f.read()
    prompt = prompt_template.replace("{slow_kernel}", config.slow_kernel)
    prompt = prompt.replace("{fast_kernel}", config.fast_kernel)
    return prompt

async def sample_once(config: SummarizerPromptConfig, agent, name):
    try:
        user_prompt = construct_summarizer_prompt(config)
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
        result = await retry_runner_safer(agent, user_prompt, run_config=run_config)
        if result is None:
            print(f"[Skip] Failed to get result for {name}")
            return
        return result
    except Exception as e:
        print(f"[Error] Failed to process {name}: {e}")
        traceback.print_exc()
        raise

async def process_optimization_item(executor_result, k, v, args, summarizer_agent):
    """Process a single optimization item summarizer"""
    service_name = executor_result["service_name"]
    case_name = executor_result["case_name"]
    
    try:
        # Determine fast and slow kernels based on speedup
        if v["speedup"] < 1.0:
            fast_kernel, slow_kernel = v["baseline"], v["body"]
        else:
            fast_kernel, slow_kernel = v["body"], v["baseline"]
        
        # Run summarizer
        prompt_config = SummarizerPromptConfig(
            user_template_path=args.user_template_path,
            slow_kernel=slow_kernel,
            fast_kernel=fast_kernel
        )
        result = await sample_once(prompt_config, summarizer_agent, f"summarizer_execution_{service_name}_{k}")
        if result is None:
            print(f"[Skip] Failed to get result for {service_name}_{k}")
            return None
        
        # Create base optimization menu item
        optimization_menu_item = {
            "service_name": service_name,
            "case_name": case_name,
            "id": k,
            "speedup_original": v["speedup"],
            "speedup": max(v["speedup"], 1.0 / v["speedup"]),
            "slow_kernel": slow_kernel,
            "fast_kernel": fast_kernel,
            "raw_response": result.raw_responses[0].output[0].summary[0].text
        }
        
        # Check if no optimization found
        if "No optimization found" in result.final_output:
            optimization_menu_item["title"] = "**No optimization found**"
            return optimization_menu_item
        else:
            optimization_menu_item["summary"] = result.final_output
            # Try to extract the title from the summary
            title = re.search(r'\*\*(.*?)\*\*', result.final_output)
            optimization_menu_item["title"] = title.group(0) if title else ""
        
        return optimization_menu_item
        
    except Exception as e:
        print(f"[Error] Failed to process {service_name}_{k}: {e}")
        traceback.print_exc()
        return None

async def main(args):
    with open(args.executor_results_path, "r") as f:
        executor_results = json.load(f)

    with open(args.model_config_path, "r") as f:
        model_config = json.load(f)

    summarizer_base_url = model_config['url']
    API_KEY = model_config['api_key']

    summarizer_client = AsyncOpenAI(base_url=summarizer_base_url, api_key=API_KEY, timeout=10000000)
    summarizer_model = OpenAIChatCompletionsModel(
        model=model_config['model'],
        openai_client=summarizer_client
    )
    set_tracing_disabled(disabled=True)

    with open(args.base_prompt_path, "r") as f:
        summarizer_system_prompt = f.read()

    summarizer_agent = Agent(
        name="Summarizer",
        instructions=summarizer_system_prompt,
        model=summarizer_model
    )

    # Remove output_list_path if it exists
    if os.path.exists(args.output_list_path):
        os.remove(args.output_list_path)

    case_name_to_service_name_plan_ids_speedups = {}
    for executor_result in executor_results["executor_results"]:
        case_name = executor_result["case_name"]
        service_name = executor_result["service_name"]
        plan_id_to_speedups = {}
        for k, v in executor_result.items():
            if k in ["service_name", "case_name"]:
                continue
            if "error" in v.keys():
                continue
            if v["speedup"] is None:
                continue
            # k=plan_{plan_id}_{sample_id}. Extract plan_id and sample_id
            plan_id, sample_id = k.split("_")[1:]
            if plan_id in plan_id_to_speedups:
                plan_id_to_speedups[plan_id]["sample_ids"].append(sample_id)
                plan_id_to_speedups[plan_id]["speedups"].append(v["speedup"])
            else:
                plan_id_to_speedups[plan_id] = {
                    "sample_ids": [sample_id],
                    "speedups": [v["speedup"]]
                }
        if case_name not in case_name_to_service_name_plan_ids_speedups:
            case_name_to_service_name_plan_ids_speedups[case_name] = {}
        for plan_id, speedups in plan_id_to_speedups.items():
            # For the same plan_id, we select the sample with the highest speedup if there is speedup > 1.0, otherwise we select the sample with the lowest speedup
            # Store all the selected "plan_{plan_id}_{sample_id}" in service_name_to_speedups[service_name]
            if max(speedups["speedups"]) > 1.0:
                cur_id = speedups['speedups'].index(max(speedups['speedups']))
                case_name_to_service_name_plan_ids_speedups[case_name][(service_name, f"plan_{plan_id}_{speedups['sample_ids'][cur_id]}")] = speedups['speedups'][cur_id]

            elif max(speedups["speedups"]) < 1.0:
                cur_id = speedups['speedups'].index(min(speedups['speedups']))
                case_name_to_service_name_plan_ids_speedups[case_name][(service_name, f"plan_{plan_id}_{speedups['sample_ids'][cur_id]}")] = speedups['speedups'][cur_id]

# Heuristic starts here:
    # Collect all plans across all services
    all_positive_plans = []  # (case_name, service_name_plan_id, speedup) for speedup > 1
    all_negative_plans = []  # (case_name, service_name_plan_id, speedup) for speedup <= 1
    
    for case_name, service_name_plan_id_speedups in case_name_to_service_name_plan_ids_speedups.items():
        for service_name_plan_id, speedup in service_name_plan_id_speedups.items():
            if speedup > 1.0:
                all_positive_plans.append((case_name, service_name_plan_id, speedup))
            else:
                all_negative_plans.append((case_name, service_name_plan_id, speedup))
    
    # Sort positive plans by speedup (descending - highest first)
    all_positive_plans.sort(key=lambda x: x[2], reverse=True)
    
    # Sort negative plans by speedup (ascending - lowest first)
    all_negative_plans.sort(key=lambda x: x[2])
    
    # Filter positive plans by threshold
    positive_plans_filtered = [(case_name, service_name_plan_id, speedup) for case_name, service_name_plan_id, speedup in all_positive_plans if speedup > args.max_threshold]
    
    # Select up to k/2 positive plans globally
    max_positive = min(len(positive_plans_filtered), args.topk // 2)
    selected_positive = positive_plans_filtered[:max_positive]
    
    # Filter negative plans by threshold
    negative_plans_filtered = [(case_name, service_name_plan_id, speedup) for case_name, service_name_plan_id, speedup in all_negative_plans if speedup < 1 / args.min_threshold]

    # Calculate remaining slots for negative plans
    remaining_slots = min(args.topk - len(selected_positive), len(negative_plans_filtered))
    selected_negative = negative_plans_filtered[:remaining_slots]
    
    # Combine all selected plans
    all_selected_plans = selected_positive + selected_negative
    
    # Group by service name
    service_name_to_selected_plan_ids = {}
    service_name_to_selected_speedups = {}
    
    for _, service_name_plan_id, speedup in all_selected_plans:
        service_name, plan_id = service_name_plan_id
        if service_name not in service_name_to_selected_plan_ids:
            service_name_to_selected_plan_ids[service_name] = []
            service_name_to_selected_speedups[service_name] = []
        
        service_name_to_selected_plan_ids[service_name].append(plan_id)
        service_name_to_selected_speedups[service_name].append(speedup)


    with open(args.output_plan_ids_path, "w") as f:
        json.dump(service_name_to_selected_plan_ids, f, indent=4)

    with open(args.output_speedups_path, "w") as f:
        json.dump(service_name_to_selected_speedups, f, indent=4)

    async with aiohttp.ClientSession() as session:
        # Collect all tasks for parallel execution
        tasks = []
        
        for executor_result in executor_results["executor_results"]:
            service_name = executor_result["service_name"]
            logfire.configure(service_name=service_name)
            logfire.instrument_openai()
            if service_name not in service_name_to_selected_plan_ids:
                continue
            for k in service_name_to_selected_plan_ids[service_name]:
                v = executor_result[k]
                task = process_optimization_item(executor_result, k, v, args, summarizer_agent)
                tasks.append(task)
        
        # Execute all tasks in parallel
        print(f"Processing {len(tasks)} optimization items in parallel...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None results and exceptions, then build the final list
        optimization_menu_list = []
        for result in results:
            if isinstance(result, Exception):
                print(f"[Error] Task failed with exception: {result}")
                continue
            if result is not None:
                optimization_menu_list.append(result)
        
        # Write all results to output file at the end
        with open(args.output_list_path, "w") as f:
            json.dump(optimization_menu_list, f, indent=4)
        
        print(f"Successfully processed {len(optimization_menu_list)} optimization items")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--executor_results_path", type=str, required=True)
    parser.add_argument("--base_prompt_path", type=str, required=True)
    parser.add_argument("--user_template_path", type=str, required=True)
    parser.add_argument("--output_list_path", type=str, required=True)
    parser.add_argument("--max_threshold", type=float, required=True)
    parser.add_argument("--min_threshold", type=float, required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--output_plan_ids_path", type=str, required=True)
    parser.add_argument("--output_speedups_path", type=str, required=True)
    parser.add_argument("--model_config_path", type=str, required=True)
    args = parser.parse_args()

    asyncio.run(main(args))