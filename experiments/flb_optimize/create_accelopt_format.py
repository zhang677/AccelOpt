import pandas as pd

candidate_input_path = "/home/ubuntu/AccelOpt/experiments/flb_optimize/partial_selected_traces_triton.csv"
candidate_output_path = "/home/ubuntu/AccelOpt/experiments/flb_full_complete_local/candidates.csv"
profile_input_path = "/home/ubuntu/AccelOpt/experiments/flb_optimize/partial_profiled_baselines.csv"
profile_output_path = "/home/ubuntu/AccelOpt/experiments/flb_full_complete_local/profile_results.csv"

df_candidates = pd.read_csv(candidate_input_path)
output_rows = []
for _, row in df_candidates.iterrows():
    output_rows.append({
        "last_solution_path": row["solution_path"],
        **row
    })
df_output = pd.DataFrame(output_rows)
df_output.to_csv(candidate_output_path, index=False)

df_profiles = pd.read_csv(profile_input_path)
output_rows = []
for _, row in df_candidates.iterrows():
    output_rows.append({
        "last_solution_path": row["solution_path"],
        **row
    })
df_output = pd.DataFrame(output_rows)
df_output.to_csv(profile_output_path, index=False)