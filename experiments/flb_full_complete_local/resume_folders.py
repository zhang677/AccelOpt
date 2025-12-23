import pandas as pd
import os
from datetime import datetime
from zoneinfo import ZoneInfo
LA = ZoneInfo("America/Los_Angeles")
from flashinfer_bench import Solution, Trace
from flashinfer_bench.data import load_json_file
from accelopt.flb_wrapper import get_unique_trace_name

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--project_name", type=str, required=True)
parser.add_argument("--org_name", type=str, required=True)
parser.add_argument("--exp_date_base", type=str, required=True)
parser.add_argument("--traceset_root", type=str, required=True)
args = parser.parse_args()
project_name = args.project_name
org_name = args.org_name
exp_date_base = args.exp_date_base

ITERS=15
BREADTH=4
TOPK_CANDIDATES=2
NUM_SAMPLES=2
MAX_THRESHOLD=1.04
MIN_THRESHOLD=1.15
TOPK=8
EXP_N=16

config_dirs=[
    "./configs",
]

proxy_profile_results_file="profile_results.csv"
proxy_profile_results_df = pd.read_csv(proxy_profile_results_file)


exp_base_dir = f"../checkpoints/{exp_date_base}"
exp_base_dir = os.path.abspath(exp_base_dir)
ACCELOPT_BASE_DIR = os.getenv("ACCELOPT_BASE_DIR")
single_loop_exec = os.path.join(ACCELOPT_BASE_DIR, "templates", "flb", "complete_local", "resume_single_loop.sh")

first_exp_date = datetime.now(LA).strftime("%m-%d-%H-%M")
for index, row in proxy_profile_results_df.iterrows():
    if not "gemm" in row["workload_path"]:
        continue
    print(f"Processing index {index}")
    service_name = get_unique_trace_name(
        load_json_file(Solution, row["solution_path"]),
        load_json_file(Trace, row["workload_path"])
    )
    
    new_exp_base_dir = os.path.join(exp_base_dir, service_name)

    with open(os.path.join(new_exp_base_dir, "log.txt"), "r") as f:
        eval_first_exp_date = f.readlines()[-1].strip()
    new_exp_config_dir = os.path.join(new_exp_base_dir, "configs")
    eval_prefix = f"eval-{index}-{exp_date_base}"

    cur_single_loop_exec_path = os.path.join(exp_base_dir, f"resume_single_loop_{service_name}.sh")
    
    with open(single_loop_exec, "r") as f:
        content = f.read()
        content = content.replace("$10", str(TOPK_CANDIDATES))
        content = content.replace("$11", str(NUM_SAMPLES))
        content = content.replace("$12", str(MAX_THRESHOLD))
        content = content.replace("$13", str(MIN_THRESHOLD))
        content = content.replace("$14", str(TOPK))
        content = content.replace("$15", str(EXP_N))
        content = content.replace("$1", f"\"{new_exp_base_dir}\"")
        content = content.replace("$2", f"\"{eval_first_exp_date}\"")
        content = content.replace("$3", f"\"{eval_prefix}\"")
        content = content.replace("$4", f"\"{project_name}\"")
        content = content.replace("$5", f"\"{org_name}\"")
        content = content.replace("$6", str(ITERS))
        content = content.replace("$7", str(BREADTH))
        content = content.replace("$8", f"\"{args.traceset_root}\"")
        with open(cur_single_loop_exec_path, "w") as f:
            f.write(content)
    os.chmod(cur_single_loop_exec_path, 0o755)