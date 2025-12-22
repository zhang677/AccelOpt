import json
import pandas as pd
import os
from accelopt.utils import init_service_name

def get_branch_id(plan_name):
    return plan_name.split("_")[1]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--executor_results_path", type=str, required=True)
    parser.add_argument("--output_base_path", type=str, required=True)
    parser.add_argument("--topk", type=int, default=1)
    args = parser.parse_args()
    with open(args.executor_results_path, "r") as f:
        executor_results = json.load(f)

    candidates = {}
    for record in executor_results["executor_results"]:
        service_name = record["baseline_solution_path"]
        case_name = record["workload_path"]
        for k, v in record.items():
            if k in ["baseline_solution_path", "workload_path"]: 
                continue
            if "error" in v.keys() and v["error"] == "No implementation found":
                continue
            branch_id = get_branch_id(k)
            if not "speedup" in v.keys() or v["speedup"] is None:
                candidates.setdefault(case_name, {}).setdefault((service_name, branch_id), []).append({
                    "definition_path": v["definition_path"],
                    "workload_path": v["workload_path"],
                    "solution_path": v["baseline_solution_path"],
                    "trace_path": v["baseline_trace_path"],
                    "latency": v["baseline_latency"],
                    "plan_id": k,
                    "priority": float("inf"),
                })
            else:
                candidates.setdefault(case_name, {}).setdefault((service_name, branch_id), []).append({
                    "definition_path": v["definition_path"],
                    "workload_path": v["workload_path"],
                    "solution_path": v["solution_path"],
                    "trace_path": v["trace_path"],
                    "latency": v["latency"],
                    "plan_id": k,
                    "priority": v["latency"],
                })
            
    # First select the best representative for each (service_name, branch_id)
    unique_candidates = {}
    for case_name, service_name_branch_id_candidates in candidates.items():
        for (service_name, branch_id), candidates_items in service_name_branch_id_candidates.items():
            best_candidate = min(candidates_items, key=lambda x: x["priority"])
            unique_candidates.setdefault(case_name, {})[(service_name, branch_id)] = best_candidate
    # Then select the topk candidates for each case
    topk_candidates = {}
    for case_name, service_name_branch_id_candidates in unique_candidates.items():
        sorted_keys = sorted(service_name_branch_id_candidates.keys(), key=lambda x: service_name_branch_id_candidates[x]["priority"])
        topk_candidates[case_name] = [service_name_branch_id_candidates[k] for k in sorted_keys[:args.topk]]
    # For candidates, store the body and spec_code into new .py files and store the service_name,task,kernel to a new csv file
    output_base_path = args.output_base_path
    os.makedirs(output_base_path, exist_ok=True)
    output_dict = []
    for case_name, candidates_items in topk_candidates.items():
        for item in candidates_items:
            output_dict.append({
                "definition_path": item["definition_path"],
                "workload_path": item["workload_path"],
                "solution_path": item["solution_path"],
                "trace_path": item["trace_path"],
            })
    output_path = f"{output_base_path}/profile_results.csv"
    output_df = pd.DataFrame(output_dict)
    output_df.to_csv(output_path, index=False)