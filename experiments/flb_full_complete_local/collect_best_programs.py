import pandas as pd
from flashinfer_bench import Solution, Trace
from flashinfer_bench.data import load_json_file
from accelopt.flb_wrapper import get_unique_trace_name
import shutil

def float_to_str(f: float) -> str:
    return f"{f:.3f}".replace(".", "dot")

exp_base_date = "12-21-17-05"
best_plan_id_path = f"/home/ubuntu/AccelOpt/experiments/flb_full_complete_local/results/{exp_base_date}/{exp_base_date.replace('-', '')}_best_plan_id.csv"
profile_results_path = "/home/ubuntu/AccelOpt/experiments/flb_full_complete_local/profile_results.csv"

best_plan_id_df = pd.read_csv(best_plan_id_path)
profile_results_df = pd.read_csv(profile_results_path)

for index, row in profile_results_df.iterrows():
    baseline_trace_path = row["trace_path"]
    matching_rows = best_plan_id_df[best_plan_id_df["baseline_trace_path"] == baseline_trace_path]
    if matching_rows.empty:
        print(f"No matching best plan ID found for trace path: {baseline_trace_path}")
        continue
    baseline_code = load_json_file(Solution, row["solution_path"]).sources[0].content
    best_code = load_json_file(Solution, matching_rows.iloc[0]["solution_path"]).sources[0].content
    unique_trace_name = get_unique_trace_name(
        load_json_file(Solution, row["solution_path"]),
        load_json_file(Trace, row["workload_path"])
    )
    baseline_latency = load_json_file(Trace, baseline_trace_path).evaluation.performance.latency_ms
    opt_latency = load_json_file(Trace, matching_rows.iloc[0]["trace_path"]).evaluation.performance.latency_ms
    with open(f"./results/{exp_base_date}/{unique_trace_name}_baseline_{float_to_str(baseline_latency)}.py", "w") as f:
        f.write(baseline_code)

    with open(f"./results/{exp_base_date}/{unique_trace_name}_best_{float_to_str(opt_latency)}.py", "w") as f:
        f.write(best_code)

    # cp "solution_path" to ./results/{exp_base_date}/{unique_trace_name}_solution.json
    shutil.copyfile(
        matching_rows.iloc[0]["solution_path"],
        f"./results/{exp_base_date}/{unique_trace_name}_solution.json"
    )