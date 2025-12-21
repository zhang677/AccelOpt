from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, RunResult, set_tracing_disabled, RunConfig, ModelSettings
import pandas as pd
from pydantic import BaseModel
import asyncio
import logfire
from datetime import datetime, timezone
from accelopt.utils import retry_runner_safer
from accelopt.flb_wrapper import format_definition
import json
from typing import List
import random
from flashinfer_bench import Definition, Solution, Trace
from flashinfer_bench.data import load_json_file

class UserPromptConfig(BaseModel):
    definition_path: str = ""
    solution_path: str = ""
    profile_str: str = ""
    prompt_template_path: str = ""
    displayed_profiles_path: str = ""
    breadth: int = 0

class PlannerResponse(BaseModel):
    service_name: str
    reasonings: List[str]
    plans: List[str]

def construct_profile_str(trace_path: str, displayed_profile_path):
    profile_str = ""
    if displayed_profile_path is not None:
        with open(displayed_profile_path, "r") as f:
            displayed_profiles = json.load(f)
    else:
        displayed_profiles = ["latency_ms"]
    profile_record_dict = {"latency_ms": load_json_file(Trace, trace_path).evaluation.performance.latency_ms}
    # shuffle the displayed_profiles
    profile_record_keys = list(profile_record_dict.keys())
    random.shuffle(profile_record_keys)
    for profile in profile_record_keys:
        if profile in displayed_profiles:
            profile_str += f"{profile}: {profile_record_dict[profile]}\n"

def construct_user_prompt(user_prompt_config: UserPromptConfig):
    with open(user_prompt_config.prompt_template_path, "r") as f:
        prompt_template = f.read()
    definition = load_json_file(Definition, user_prompt_config.definition_path)
    solution = load_json_file(Solution, user_prompt_config.solution_path)
    user_prompt = prompt_template.replace("{definition_str}", format_definition(definition))
    user_prompt = user_prompt.replace("{kernel_code}", solution.sources[0].content)
    user_prompt = user_prompt.replace("{profile}", user_prompt_config.profile_str)
    return user_prompt

def seperate_reasoning(result: RunResult | None):
    if result is None:
        return None, None
    plan = result.final_output
    try:
        reasoning = result.raw_responses[0].output[0].summary[0].text
    except:
        reasoning = plan
    return reasoning, plan

async def single_query(single_record, agent, user_prompt_config: UserPromptConfig):
    config_copy = user_prompt_config.model_copy()
    profile_str = construct_profile_str(single_record["trace_path"], config_copy.displayed_profiles_path)
    config_copy.profile_str = profile_str
    config_copy.definition_path = single_record["definition_path"]
    config_copy.solution_path =single_record["solution_path"]
    user_prompt = construct_user_prompt(config_copy)
    logfire_service_name = load_json_file(Solution, single_record["service_name"]).name
    logfire.configure(service_name=logfire_service_name)
    logfire.instrument_openai()
    with logfire.span(logfire_service_name) as span:
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
        results = await asyncio.gather(*[retry_runner_safer(agent, user_prompt, run_config=run_config) for _ in range(user_prompt_config.breadth)])
        if results is None:
            return None
        reasonings = []
        plans = []
        for result in results:
            reasoning, plan = seperate_reasoning(result)
            if reasoning is not None:
                reasonings.append(reasoning)
                plans.append(plan)
        return PlannerResponse(
            service_name=single_record["service_name"],
            reasonings=reasonings,
            plans=plans
        )

async def main(kwargs):
    agent = kwargs["agent"]
    user_prompt_config = kwargs["user_prompt_config"]
    record_data = kwargs["record_data"]
    return await single_query(record_data, agent, user_prompt_config)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--breadth", type=int, required=True)
    parser.add_argument("--exp_dir", type=str, required=True)
    parser.add_argument("--base_prompt_path", type=str, required=True)
    parser.add_argument("--user_template_path", type=str, required=True)
    parser.add_argument("--profile_result_path", type=str, required=True)
    parser.add_argument("--model_config_path", type=str, required=True)
    parser.add_argument("--displayed_profiles_path", type=str, required=False, default=None)
    args = parser.parse_args()

    profile_result_path = args.profile_result_path
    base_prompt_path = args.base_prompt_path
    user_template_path = args.user_template_path
    output_path = args.output_path
    breadth = args.breadth
    displayed_profiles_path = args.displayed_profiles_path
    exp_dir = args.exp_dir
    # Start timing
    current_time = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    time_record_path = f"{exp_dir}/planner_start_end_time.txt"
    with open(time_record_path, "w") as f:
        f.write(f"{current_time},")

    df = pd.read_csv(profile_result_path)
    row_data_list = df.to_dict(orient="records")
    

    with open(args.model_config_path, "r") as f:
        model_config = json.load(f)
    
    BASE_URL = model_config['url']
    API_KEY = model_config['api_key']

    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=10000000)
    model = OpenAIChatCompletionsModel(
        model=model_config["model"],# "openai/gpt-oss-120b",
        openai_client=client
    )

    agent = Agent(
        name="Planner",
        instructions=open(base_prompt_path, "r").read(),
        model=model
    )

    # Prepare all arguments
    all_main_args = []
    results = []
    for row_data in row_data_list:
        main_arg = {
            "agent": agent,
            "user_prompt_config": UserPromptConfig(
                prompt_template_path=user_template_path,
                breadth=breadth,
                displayed_profiles_path=displayed_profiles_path
            ),
            "record_data": row_data
        }
        all_main_args.append(main_arg)

    set_tracing_disabled(disabled=True)
    
    # Run all queries concurrently without limits
    async def run_all():
        tasks = [main(kwargs) for kwargs in all_main_args]
        return await asyncio.gather(*tasks)

    results = asyncio.run(run_all())
    
    # Filter out None results and exceptions, convert to dicts
    results_list = []
    for result in results:
        if result is not None and not isinstance(result, Exception):
            if hasattr(result, 'dict'):
                results_list.append(result.model_dump())
            else:
                results_list.append(result)
    
    # Save results
    json.dump(results_list, open(output_path, "w"), indent=4)

    # End timing
    current_time = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    with open(time_record_path, "a") as f:
        f.write(f"{current_time}")