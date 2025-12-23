import os
import numpy as np
import json
import matplotlib.pyplot as plt
from accelopt.flb_plotting_utils import (
    get_baseline_latency, get_exp_dates, 
    create_plot_figure, series_from_with_metadata, get_avg_good_speedup
)
import pandas as pd
from flashinfer_bench import Trace, Solution
from flashinfer_bench.data import load_json_file
from accelopt.flb_wrapper import get_unique_trace_name
from pathlib import Path


exp_base_dir = "../checkpoints/12-21-17-05"
output_path = "./results/12171705.png"
baseline_result_path = "profile_results.csv"
exp_title = "TopK=8, ExpN=16, B=2, N=4, K=2, T=16, gpt-oss-120b, gpt-oss-120b"
# Load baseline
baseline_df = pd.read_csv(baseline_result_path)
# baseline_trace_paths = baseline_df["solution_path"].tolist()
# List of tuple (trace_path, solution_path)
baseline_solution_workload_trace_paths = list(zip(baseline_df["solution_path"].tolist(), baseline_df["workload_path"].tolist(), baseline_df["trace_path"].tolist()))
# Create figure and axes
fig, axes = create_plot_figure(baseline_solution_workload_trace_paths)

# Plot each case
plan_id_rows = []
best_plan_id_rows = []
for ax, (baseline_solution_path, workload_path, baseline_trace_path) in zip(axes, baseline_solution_workload_trace_paths):
    baseline_latency = get_baseline_latency(baseline_df, baseline_trace_path)
    workload = load_json_file(Trace, baseline_trace_path)
    solution = load_json_file(Solution, baseline_solution_path)
    exp_dir = os.path.join(exp_base_dir, get_unique_trace_name(solution, workload))
    exp_dates = get_exp_dates(exp_dir)
    exp_vals, exp_trace_paths, exp_best_plan_keys = series_from_with_metadata(exp_dir, exp_dates, workload_path, baseline_latency, "workload_path")
    avg_good_speedup = get_avg_good_speedup(exp_dir, exp_dates, workload_path, baseline_latency, "workload_path")
    
    max_len = max(len(exp_vals), 1)
    x_exp = np.arange(len(exp_vals))

    l0, = ax.plot(x_exp, exp_vals, marker="o", label=exp_title)
    ax.set_ylabel("Best Speedup")
    ax.set_ylim(-0.05, 2.1)
    ax.set_title(Path(baseline_trace_path).stem)
    ax.set_xticks(np.arange(max_len))
    ax.set_xlabel("Test Iteration")
    ax.legend(loc="best")
    

    # Get the length of the aggregated rewrites
    generated_rewrites_lens = []
    for d in exp_dates:
        aggregated_rewrites_path = os.path.join(exp_dir, d, "rewrites", "aggregated_rewrites_list.json")
        if not os.path.exists(aggregated_rewrites_path):
            continue
        with open(aggregated_rewrites_path, "r") as f:
            aggregated_rewrites = json.load(f)
        generated_rewrites_lens.append(len(aggregated_rewrites))
    ax2 = ax.twinx()
    l2, = ax2.plot(x_exp, generated_rewrites_lens[:len(x_exp)], "o--", color="green", label="aggregated_rewrites_length")
    ax2.set_ylabel("Aggregated Rewrites Length")
    ax2.set_ylim(-0.05, 16.05)
    h1, lab1 = ax.get_legend_handles_labels()
    h2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, lab1 + lab2, loc="best")

    if len(exp_vals) > 0:
        best_idx = int(np.nanargmax(np.round(exp_vals, 2)))
        best_val = exp_vals[best_idx]
        ax.scatter([best_idx], [best_val], s=80, facecolors="none", edgecolors="red", linewidths=2, zorder=5)
        ax.annotate(f"{best_val:.2f}", xy=(best_idx, best_val), xytext=(0, 8),
                    textcoords="offset points", ha="center", va="bottom", fontsize=16,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
        best_plan_id_rows.append({
            "baseline_trace_path": baseline_trace_path,
            "experiment":exp_title,
            "iter": best_idx,
            "plan_id": exp_best_plan_keys[best_idx],
            "exp_date": exp_dates[best_idx],
            "trace_path": exp_trace_paths[best_idx],
            "best_speedup": best_val,
        })

    for i, pid in enumerate(exp_best_plan_keys):
        plan_id_rows.append({
            "baseline_trace_path": baseline_trace_path,
            "experiment": exp_title,
            "iter": i,
            "plan_id": pid,
            "exp_date": exp_dates[i],
            "best_speedup": exp_vals[i],
            "avg_good_speedup": avg_good_speedup[i],
            "trace_path": exp_trace_paths[i],
        })

# ---------- Plan ID tables ----------
os.makedirs(os.path.dirname(output_path), exist_ok=True)

plan_ids_long = pd.DataFrame(plan_id_rows).sort_values(["baseline_trace_path", "experiment", "iter"])
long_csv = output_path.replace(".png", "_plan_ids_long.csv")
plan_ids_long.to_csv(long_csv, index=False)
print(f"Plan IDs (long) saved to: {long_csv}")

best_plan_id = pd.DataFrame(best_plan_id_rows).sort_values(["baseline_trace_path", "experiment", "iter"])
best_csv = output_path.replace(".png", "_best_plan_id.csv")
best_plan_id.to_csv(best_csv, index=False)
print(f"Best plan IDs saved to: {best_csv}")

from scipy.stats import gmean
geomean_best_speedup = gmean(best_plan_id["best_speedup"])
print(f"Geomean best speedup: {geomean_best_speedup}")

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Plot saved to: {output_path}")