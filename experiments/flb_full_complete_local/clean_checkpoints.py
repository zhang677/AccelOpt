import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--chkpt_path", type=str, required=True)
args = parser.parse_args()
chkpt_path = args.chkpt_path

import os
import shutil
import pandas as pd
from flashinfer_bench import Solution, Trace
from flashinfer_bench.data import load_json_file
from accelopt.flb_wrapper import get_unique_trace_name
proxy_profile_results_file="profile_results.csv"
proxy_profile_results_df = pd.read_csv(proxy_profile_results_file)
case_names = []
for index, row in proxy_profile_results_df.iterrows():
    service_name = get_unique_trace_name(
        load_json_file(Solution, row["solution_path"]),
        load_json_file(Trace, row["workload_path"])
    )
    case_names.append(service_name)

exp_base_dir = args.chkpt_path
for case_name in case_names:
    case_dir = os.path.join(exp_base_dir, case_name)
    with open(os.path.join(case_dir, "log.txt"), "r") as f:
        log_lines = f.readlines()
    exp_dates = [line.strip() for line in log_lines]
    exp_date_to_delete = []
    exp_date_to_keep = []
    for exp_date in exp_dates:
        executor_results_path = os.path.join(case_dir, exp_date, "executor_results.json")
        if not os.path.exists(executor_results_path):
            exp_date_to_delete.append(exp_date)
            continue
        with open(executor_results_path, "r") as f:
            executor_result_str = f.read()
            if executor_result_str.count("No implementation found") > 1:
                exp_date_to_delete.append(exp_date)
                continue
        exp_date_to_keep.append(exp_date)
    for exp_date_del in exp_date_to_delete:
        exp_date_dir = os.path.join(case_dir, exp_date_del)
        if os.path.exists(exp_date_dir):
            shutil.rmtree(exp_date_dir)
    with open(os.path.join(case_dir, "log.txt"), "w") as f:
        for exp_date in exp_date_to_keep:
            f.write(exp_date + "\n")
    print(f"{case_name}: exps to delete: {exp_date_to_delete}; recover from: {exp_date_to_keep[-1]}")
