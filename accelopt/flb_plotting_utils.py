import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Optional
from flashinfer_bench import Trace
from flashinfer_bench.data import load_json_file

def get_baseline_latency(baseline_df: pd.DataFrame, baseline_trace_path: str) -> float:
    row = baseline_df[baseline_df["trace_path"] == baseline_trace_path]
    if row.empty:
        raise RuntimeError(f"Baseline latency not found for trace_path={baseline_trace_path}")
    return load_json_file(Trace, row["trace_path"].iloc[0]).evaluation.performance.latency_ms

def get_exp_dates(exp_dir: str, max_iters: int) -> List[str]:
    log_file = os.path.join(exp_dir, "log.txt")
    with open(log_file, "r") as f:
        exp_dates = f.readlines()
    exp_dates = [d.strip() for d in exp_dates]
    return exp_dates[:max_iters]

def create_plot_figure(case_names: List[str], figsize: Tuple[int, int] = None) -> Tuple[plt.Figure, List[plt.Axes]]:
    n = len(case_names)
    if figsize is None:
        figsize = (8, 2.6 * n)
    
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharey=True)
    if n == 1:
        axes = [axes]
    
    return fig, axes

def get_sample_id(plan_id: str) -> int:
    return int(plan_id.split("_")[2])

def get_plan_id(plan_id: str) -> str:
    return int(plan_id.split("_")[1])

def get_latency_dict(record: dict, max_plans: int = None, max_sample_id: int = None) -> Dict[Tuple[str, str], Dict[str, Any]]:
    d = {}
    service_name = record["baseline_solution_path"]
    for k, v in record.items():
        if isinstance(v, dict) and "latency" in v and v["latency"] is not None:
            sample_id = get_sample_id(k)
            if max_sample_id is not None and sample_id >= max_sample_id:
                continue
            plan_id_int = get_plan_id(k)
            if max_plans is not None and plan_id_int >= max_plans:
                continue
            d[(service_name, k)] = {
                "latency": v["latency"],
                "trace_path": v["trace_path"],
                "solution_path": v["solution_path"],
                "plan_id": k
            }
    return d

def calc_best_metadata_for_file(executor_results_json: str, baseline_latency: float, 
                              target_case: str, key_name: str, max_candidates: int = None, max_plans: int = None, max_sample_id: int = None) -> Tuple[Optional[float], Optional[Tuple[str, str]]]:
    if not os.path.exists(executor_results_json):
        return None, None, None
    
    with open(executor_results_json, "r") as f:
        data = json.load(f)
    
    results = data.get("executor_results", [])
    complete_lat_dict = {}
    
    cur_num_candidates = 0
    for r in results:
        if r.get(key_name) != target_case:
            print("Skipping", r.get(key_name), "not equal to", target_case)
            continue
        lat_dict = get_latency_dict(r, max_plans, max_sample_id)
        complete_lat_dict.update(lat_dict)
        cur_num_candidates += 1
        if max_candidates is not None and cur_num_candidates > max_candidates:
            break

    best = 0.0
    best_plan_key = None
    best_metadata = None
    
    for item_key, lat in complete_lat_dict.items():
        if lat["latency"] > 0:
            sp = baseline_latency / lat["latency"]
            if sp > best:
                best = sp
                best_metadata = {
                    "trace_path": lat["trace_path"],
                    "solution_path": lat["solution_path"]
                }
                best_plan_key = item_key
    
    return (best, best_metadata, best_plan_key) if best > 0 else (0.0, {}, None)

def series_from_with_metadata(base_dir: str, dates: List[str], case_name: str, 
                baseline_latency: float, key_name: str, max_candidates: int = None, max_plans: int = None, max_sample_id: int = None) -> Tuple[List[float], List[Optional[Tuple[str, str]]]]:
    vals = []
    best_plan_keys = []
    metadatas = []
    for d in dates:
        path = os.path.join(base_dir, d, "executor_results.json")
        best, best_metadata, best_plan_key = calc_best_metadata_for_file(path, baseline_latency, case_name, key_name, max_candidates, max_plans, max_sample_id)
        if best is not None:
            vals.append(best)
            metadatas.append(best_metadata)
            best_plan_keys.append(best_plan_key)
    
    return vals, metadatas, best_plan_keys

def get_avg_good_speedup(base_dir: str, dates: List[str], case_name: str, 
                baseline_latency: float, key_name: str, max_candidates: int = None, max_plans: int = None, max_sample_id: int = None, good_speedup_thresholds = [1.00, 1.01, 1.05, 1.10, 1.20, 1.40]) -> List[Dict[float, int]]:
    vals = []
    for d in dates:
        path = os.path.join(base_dir, d, "executor_results.json")
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            data = json.load(f)
        results = data.get("executor_results", [])
        single_iter_vals = {
            good_speedup_threshold: 0
            for good_speedup_threshold in good_speedup_thresholds
        }
        cur_num_candidates = 0
        for r in results:
            if r.get(key_name) != case_name:
                continue
            lat_dict = get_latency_dict(r, max_plans, max_sample_id)
            for item_key, lat in lat_dict.items():
                if lat["latency"] > 0:
                    sp = baseline_latency / lat["latency"]
                    for good_speedup_threshold in good_speedup_thresholds:
                        if sp > good_speedup_threshold:
                            single_iter_vals[good_speedup_threshold] += 1
            cur_num_candidates += 1
            if max_candidates is not None and cur_num_candidates > max_candidates:
                break
        vals.append(json.dumps(single_iter_vals))
    return vals