import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--chkpt_path", type=str, required=True)
args = parser.parse_args()
chkpt_path = args.chkpt_path

import os
import shutil
case_names = [
    "adamw_M10944_N2048",
    "add_rmsnorm_matmul_K1024_M4096_N2048",
    "bmm_B16_K64_M4096_N4096",
    "bmm_softmax_B16_K64_M4096_N4096",
    "gqa_full_B1_D128_KH8_N4096_QH16",
    "lora_K5120_M4096_N12288_R128",
    "mamba_C256_M7168_S16",
    "matmul_add_rmsnorm_K2048_M4096_N2048",
    "matmul_K5120_M4096_N12288",
    "rmsnorm_matmul_K1024_M4096_N2048",
    "rope_single_freq_apply_B1_D128_H64_N4096",
    "silu_M4096_N7168",
    "swiglu_K1024_M4096_N3072",
    "transpose_matmul_K2048_M4096_N10944"
]

exp_base_dir = args.chkpt_path
for case_name in case_names:
    case_name = case_name + "_ID0"
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
