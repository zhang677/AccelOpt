# Each definition name has two bars: latency_ms of baseline and optimized
import matplotlib.pyplot as plt
import pandas as pd
from flashinfer_bench import Trace, Definition
from flashinfer_bench.data import load_json_file

exp_base_date = "12-21-17-05"
exp_title = "TopK=8, ExpN=16, B=2, N=4, K=2, T=16, gpt-oss-120b, gpt-oss-120b"
results_path = f"./results/{exp_base_date}/{exp_base_date.replace('-', '')}_best_plan_id.csv"
baseline_profiles_path = "profile_results.csv"
df = pd.read_csv(results_path)
baseline_df = pd.read_csv(baseline_profiles_path)
# 1. Initialize a list to store the extracted data
plot_data = []
fontsize = 14
for index, row in df.iterrows():
    baseline_trace_path = row["baseline_trace_path"]
    optimized_trace_path = row["trace_path"]
    
    baseline_trace = load_json_file(Trace, baseline_trace_path)
    optimized_trace = load_json_file(Trace, optimized_trace_path)
    
    baseline_latency = baseline_trace.evaluation.performance.latency_ms
    optimized_latency = optimized_trace.evaluation.performance.latency_ms
    definition_path = baseline_df[baseline_df["trace_path"] == baseline_trace_path]["definition_path"].iloc[0]
    definition_name = load_json_file(Definition, definition_path).op_type

    # 2. Append the data to our list
    plot_data.append({
        "Definition": definition_name,
        "Baseline": baseline_latency,
        "Optimized": optimized_latency
    })

# 3. Create a DataFrame for plotting
plot_df = pd.DataFrame(plot_data)

# 4. Set the Definition as the index so it appears on the X-axis
plot_df.set_index("Definition", inplace=True)
plot_df = plot_df.sort_values(by="Baseline", ascending=True)
# 5. Plot the grouped bar chart
ax = plot_df.plot(kind="bar", figsize=(12, 6), width=0.8)

# 6. Add labels and title
plt.title(exp_title, fontsize=fontsize)
plt.ylabel("Latency (ms)", fontsize=fontsize)
plt.xlabel("Kernel OpType", fontsize=fontsize)
plt.xticks(rotation=45, ha='right', fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title="Execution Type")
plt.yscale('log')

# Optional: Add data labels on top of bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 9), 
                textcoords='offset points',
                fontsize=fontsize)

plt.tight_layout()
plt.savefig(results_path.replace("_best_plan_id.csv", "_latency_comparison.png"))