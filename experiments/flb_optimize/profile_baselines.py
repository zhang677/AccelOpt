from accelopt.flb_wrapper import FlashInferKernel, get_unique_trace_name
from flashinfer_bench import Definition, Solution, Trace
from flashinfer_bench.data import save_json_file, load_json_file
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_date_base", type=str, required=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    selected_traces_df = pd.read_csv("/home/ubuntu/AccelOpt/experiments/flb_optimize/partial_selected_traces_triton.csv")
    traceset_path = "/home/ubuntu/AccelOpt/experiments/flb_interface/checkpoints"
    output_base_path = Path(f"../checkpoints/{args.exp_date_base}")
    output_rows = []
    for index, row in tqdm(selected_traces_df.iterrows()):
        def_path = row["definition_path"]
        solution_path = row["solution_path"]
        workload_path = row["workload_path"]
        kernel = FlashInferKernel(traceset_path, def_path)
        trace, res = kernel.profile(
            solution_path,
            workload_path=workload_path,
            timeout_seconds=300,
            profile_baseline=True,
            use_isolated_runner=True
        )
        definition = load_json_file(Definition, def_path)
        solution = load_json_file(Solution, solution_path)
        workload = load_json_file(Trace, workload_path)
        output_trace_path = output_base_path / "traces" / definition.op_type / (get_unique_trace_name(solution, workload) + ".json")
        save_json_file(trace, output_trace_path)
        output_rows.append({
            **row,
            "trace_path": output_trace_path
        })

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_base_path / f"profiled_baselines_{args.exp_date_base}.csv", index=False)


if __name__ == "__main__":
    main()