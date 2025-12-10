import pandas as pd
from scipy.stats import gmean
import json

PEAK_BW = 410 * 1024 * 1024 * 1024 # 410 GiB/s
PEAK_VEC_FLOPS = 2 * 128 * 1.12 * 1e9 # 286.8 GFlops (Vec + Scalar)
PEAK_MATMUL_FLOPS = 23.75 * 1e12 # 23.75 TFlops

def calc_theoretical_peak(problem_name, config):
    vec_flops = 0
    mm_flops = 0
    total_bytes = 0
    # Rules:
    # 1. Exclude ops on const scalars (because they are normally fused with other ops)
    if problem_name == "mamba":
        M = config["M"]
        C = config["C"]
        S = config["S"]
        vec_flops = M * C * S * (2 + 2) + M * (C * S * 2) + M * C * S * 2
        total_bytes = (C * M * 2 + C * S + S * M * 2 + C * M) * 4
    elif problem_name == "silu":
        M = config["M"]
        N = config["N"]
        vec_flops = M * N * 3
        total_bytes = (M * N + N * M) * 4
    elif problem_name == "add_rmsnorm_matmul":
        M = config["M"]
        N = config["N"]
        K = config["K"]
        vec_flops = M * K * 6
        mm_flops = M * N * K * 2
        total_bytes = (M * K * 2 + K * N + K + M * N) * 4
    elif problem_name == "matmul_add_rmsnorm":
        M = config["M"]
        N = config["N"]
        K = config["K"]
        vec_flops = M * N * 6
        mm_flops = M * N * K * 2
        total_bytes = (M * K + N * K + M * N + N + M * N) * 4
    elif problem_name == "rmsnorm_matmul":
        M = config["M"]
        N = config["N"]
        K = config["K"]
        vec_flops = M * K * 4
        mm_flops = M * N * K * 2
        total_bytes = (M * K + K * N + M * N) * 4
    elif problem_name == "swiglu":
        M = config["M"]
        N = config["N"]
        K = config["K"]
        vec_flops = M * N * 5
        mm_flops = M * K * N * 2 * 3
        total_bytes = (M * K * 2 + K * N * 3) * 4
    elif problem_name == "matmul":
        M = config["M"]
        N = config["N"]
        K = config["K"]
        mm_flops = M * N * K * 2
        total_bytes = (M * N + N * K + M * K) * 4
    elif problem_name == "gqa_full":
        B = config["B"]
        N = config["N"]
        QH = config["QH"]
        KH = config["KH"]
        D = config["D"]
        mm_flops = B * QH * N * D * N * 2 * 2
        vec_flops = B * QH * N * N * 5
        total_bytes = (B * QH * N * D * 2 + B * KH * N * D * 2) * 4
    elif problem_name == "rope_single_freq_apply":
        B = config["B"]
        H = config["H"]
        N = config["N"]
        D = config["D"]
        vec_flops = B * H * N * D * 3
        total_bytes = (D * B * H * N) * 3 * 4
    elif problem_name == "bmm":
        B = config["B"]
        M = config["M"]
        N = config["N"]
        K = config["K"]
        mm_flops = B * M * N * K * 2
        total_bytes = (B * M * N + B * N * K + B * M * K) * 4
    elif problem_name == "bmm_softmax":
        B = config["B"]
        M = config["M"]
        N = config["N"]
        K = config["K"]
        vec_flops = B * M * N * 4 + B * M * N
        mm_flops = B * M * N * K * 2
        total_bytes = (B * M * N + B * N * K + B * M * K) * 4
    elif problem_name == "transpose_matmul":
        M = config["M"]
        N = config["N"]
        K = config["K"]
        mm_flops = M * N * K * 2
        total_bytes = (M * N + N * K + M * K) * 4
    elif problem_name == "lora":
        M = config["M"]
        N = config["N"]
        K = config["K"]
        R = config["R"]
        vec_flops = M * N
        mm_flops = M * N * K * 2 + M * K * R * 2 + M * N * R * 2
        total_bytes = (M * K + K * N + K * R + R * N) * 4 + M * N * 4
    elif problem_name == "adamw":
        M = config["M"]
        N = config["N"]
        vec_flops = M * N * (1 + 1 + 2 + 3)
        total_bytes = M * N * 4 * 4 + M * N * 4

    else:
        raise ValueError(f"Headroom {headroom} is less than best speedup {best_speedup} or 1")
    
    latency = {
        "vec": vec_flops / PEAK_VEC_FLOPS,
        "mm": mm_flops / PEAK_MATMUL_FLOPS,
        "memory": total_bytes / PEAK_BW
    }
    bound_key = max(latency, key=latency.get)
    return {
        "theoretical_peak_latency": latency[bound_key] * 1e3,
        "bound_key": bound_key
    }

def get_baseline_latency(profile_str):
    profile = json.loads(profile_str)
    return profile["latency"]


baseline_result_path = "profile_results.csv"

records = [
    {
        "best_result_path": "../checkpoints/11-11-09-04/11110904_best_plan_id.csv",
        "output_path": "theoretical_headroom_11110904.csv"
    }
]

baseline_df = pd.read_csv(baseline_result_path)


for record in records:
    best_df = pd.read_csv(record["best_result_path"])

    output_rows = []
    for index, row in baseline_df.iterrows():
        baseline_latency = get_baseline_latency(row["profile"])
        config = json.loads(row["values"])
        best_speedup = best_df[best_df["case_name"] == row["case_name"]]["best_speedup"].values[0]
        problem_name = row["problem"]
        theoretical_peak = calc_theoretical_peak(problem_name, config)
        headroom = baseline_latency / theoretical_peak["theoretical_peak_latency"]
        if headroom < best_speedup or headroom < 1:
            raise ValueError(f"Headroom {headroom} is less than best speedup {best_speedup} or 1")
        output_rows.append({
            "problem": problem_name,
            "case_name": row["case_name"],
            "headroom": headroom,
            "baseline_percentage": 1 / headroom,
            "achieved_percentage": best_speedup / headroom,
            "best_speedup": best_speedup,
            "baseline_latency": baseline_latency,
            "theoretical_peak_latency": theoretical_peak["theoretical_peak_latency"],
            "bound_key": theoretical_peak["bound_key"]
        })

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(record["output_path"], index=False)

    headroom_mean = gmean(output_df["headroom"])
    best_speedup_mean = gmean(output_df["best_speedup"]) 
    print("Geomean baseline percentage: ", 1 / headroom_mean)
    print("Geomean achieved percentage: ", best_speedup_mean / headroom_mean)
    print("Remaining headroom: ", headroom_mean / best_speedup_mean)