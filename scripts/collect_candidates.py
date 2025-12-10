import os
import argparse
import pandas as pd
import json
from accelopt.utils import get_case_name
from accelopt.kernel_wrapper import NKIKernel

parser = argparse.ArgumentParser()
parser.add_argument("--summary_path", type=str, default="")
parser.add_argument("--output_candidates_path", type=str, required=True)
parser.add_argument("--output_profile_path", type=str, required=True)
parser.add_argument("--save_fields_path", type=str, default="")
parser.add_argument("--nc_id", type=int, default=0)
parser.add_argument("--mode", type=str, default="construct")
args = parser.parse_args()

nkibench_base_path = os.path.join(os.getenv("ACCELOPT_BASE_DIR"), "NKIBench")


output_dict = []
problems = [
    "add_rmsnorm_matmul",
    "matmul_add_rmsnorm",
    "gqa_full",
    "matmul",
    "rmsnorm_matmul",
    "rope_single_freq_apply",
    "swiglu",
    "silu",
    "bmm",
    "bmm_softmax",
    "transpose_matmul",
    "lora",
    "adamw",
    "mamba"
]

def construct_table():
    base_summary_path = os.path.join(nkibench_base_path, "summary.json")
    with open(base_summary_path, "r") as f:
        summary = json.load(f)

    output_rows = []
    for problem_name, problem_info in summary.items():
        for case_id, case_info in problem_info["cases"].items():
            for single_impl in case_info["impls"]:
                if problem_name not in problems:
                    continue
                case_name = get_case_name(problem_name, case_info["values"])
                row = {
                    "problem": problem_name,
                    "values": json.dumps(case_info["values"]),
                    "case_id": case_id,
                    "task": os.path.join(nkibench_base_path, single_impl["task"]),
                    "kernel": os.path.join(nkibench_base_path, single_impl["kernel"]),
                    "case_name": case_name,
                    "service_name": case_name + "_ID0"
                } 
                output_rows.append(row)
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(args.output_candidates_path, index=False)


def construct_profile_table():
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(args.nc_id)
    df = pd.read_csv(args.output_candidates_path)
    with open(args.save_fields_path, "r") as f:
        save_fields = json.load(f)
    output_rows = []
    for index, row in df.iterrows():
        nki_kernel = NKIKernel(row["kernel"], row["task"])
        nki_kernel.rel_tol = 3e-5 if "mamba" in row["problem"] else 2e-5
        nki_kernel.profile(save_fields)
        profile_data = {"profile": json.dumps(nki_kernel.res.metadata)}
        output_rows.append({**row, **profile_data})
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(args.output_profile_path, index=False)

if __name__ == "__main__":
    if args.mode == "construct":
        construct_table()
        construct_profile_table()
    elif args.mode == "collect":
        construct_table()
    elif args.mode == "profile":
        construct_profile_table()
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
    
    
    