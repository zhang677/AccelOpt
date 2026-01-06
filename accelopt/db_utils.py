import json
from flashinfer_bench import Definition, Trace, Solution, BenchmarkConfig, Evaluation
from supabase import Client
from hashlib import sha256
from dataclasses import fields
import logging

logger = logging.getLogger(__name__)

def try_upsert(supabase_client: Client, table_name: str, data: dict):
    conflict_columns = {
        "definitions": "name",
        "workloads": "workload_uuid",
        "solutions": "solution_hash,hash_version",
        "profiles": "workload_id,solution_id,benchmark_config_hash,eval_hardware"
    }
    
    try:
        return supabase_client.table(table_name).upsert(
            data, 
            on_conflict=conflict_columns.get(table_name, ""),
            ignore_duplicates=False
        ).execute()
        
    except Exception as e:
        logger.error(f"Database error on {table_name}: {e}")
        return None


def get_solution_hash(solution: Solution, hash_version: int = 1) -> str:
    if hash_version == 1:
        assert solution.spec.language in ["triton", "python"], "Only 'triton' and 'python' solutions are supported in this hash."
        source_code = solution.sources[0].content
        code_hash = sha256(source_code.encode('utf-8')).hexdigest()
    else:
        raise ValueError(f"Unsupported hash_version: {hash_version}")
    return code_hash

def get_benchmark_config_hash(bench_config: BenchmarkConfig) -> (tuple[dict, str]):
    # 1. Define the performance-critical fields exactly as requested
    perf_fields = {
        "warmup_runs", "iterations", "num_trials", "rtol", "atol",
        "required_matched_ratio", 
        "sampling_validation_trials", "sampling_tvd_threshold", "timeout_seconds"
    }

    # 2. Extract only those fields into a dictionary
    # Using getattr() ensures we get the current values of the instance
    perf_data = {
        f.name: getattr(bench_config, f.name) 
        for f in fields(bench_config) 
        if f.name in perf_fields
    }

    config_str = json.dumps(perf_data, sort_keys=True, separators=(',', ':'))
    config_hash = sha256(config_str.encode('utf-8')).hexdigest()
    return perf_data, config_hash

def definition_orm(obj: Definition):
    return {
        "definition": obj.model_dump(mode="json", exclude_unset=True),
    }

def workload_orm(supabase_client: Client, obj: Trace):
    def_name = obj.definition
    # Get the definition_id based on the def_name
    response = supabase_client.table("definitions").select("id").eq("name", def_name).execute()
    if response.data:
        definition_id = response.data[0]['id']
    else:
        raise ValueError(f"Definition with name '{def_name}' not found in the database.")
    return {
        "definition_id": definition_id,
        "workload": obj.model_dump(mode="json", exclude_unset=True, exclude={'solution', 'evaluation'}) # Exclude solution and evaluation
    }

def solution_orm(supabase_client: Client, obj: Solution, hash_version: int = 1):
    assert obj.spec.language in ["triton", "python"], "Only 'triton' and 'python' solutions are supported in this ORM."
    def_name = obj.definition
    # Get the definition_id based on the def_name
    response = supabase_client.table("definitions").select("id").eq("name", def_name).execute()
    if response.data:
        definition_id = response.data[0]['id']
    else:
        raise ValueError(f"Definition with name '{def_name}' not found in the database.")
    if hash_version == 1:
        code_hash = get_solution_hash(obj, hash_version=hash_version)
    else:
        raise ValueError(f"Unsupported hash_version: {hash_version}")
    return {
        "definition_id": definition_id,
        "solution": obj.model_dump(mode="json", exclude_unset=True),
        "hash_version": hash_version,
        "solution_hash": code_hash,
        "evolve_metadata": {}
    }

def profile_orm(supabase_client: Client, obj: Trace, bench_config: BenchmarkConfig, solution_id: str):
    workload_uuid = obj.workload.uuid
    response = supabase_client.table("workloads").select("id").eq("workload_uuid", workload_uuid).execute()
    if response.data:
        workload_id = response.data[0]['id']
    else:
        raise ValueError(f"Workload with uuid '{workload_uuid}' not found in the database.")
    
    perf_data, bench_config_hash = get_benchmark_config_hash(bench_config)
    # evaluation_dict = obj.evaluation.model_dump(mode="json", exclude_unset=True)
    evaluation_dict = json.loads(obj.evaluation.model_dump_json(exclude_unset=True))

    return {
        "workload_id": workload_id,
        "solution_id": solution_id,
        "evaluation": evaluation_dict,
        "benchmark_config": perf_data,
        "benchmark_config_hash": bench_config_hash
    }

def definition_to_obj(data) -> Definition:
    return Definition.model_validate(data["definition"])

def workload_to_obj(data) -> Trace:
    return Trace.model_validate(data["workload"])

def solution_to_obj(data) -> Solution:
    return Solution.model_validate(data["solution"])

def profile_to_obj(supabase_client: Client, data) -> Trace:
    workload_data = supabase_client.table("workloads").select("workload").eq("id", data["workload_id"]).execute().data[0]["workload"]
    solution = supabase_client.table("solutions").select("solution").eq("id", data["solution_id"]).execute().data[0]["solution"]["name"]
    evaluation_data = data["evaluation"]
    trace = Trace.model_validate(workload_data)
    trace.solution = solution
    trace.evaluation = Evaluation.model_validate(evaluation_data)
    return trace
