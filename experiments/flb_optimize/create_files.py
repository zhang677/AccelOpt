from flashinfer_bench import TraceSet, Trace
from flashinfer_bench.data import load_jsonl_file, save_json_file
from accelopt.flb_wrapper import get_workload_stem_name
from pathlib import Path
import pandas as pd
def find_first(list, predicate):
    for item in list:
        if predicate(item):
            return item
    return None


output_base_path = Path("/home/ubuntu/AccelOpt/experiments/flb_optimize")
traceset_path = Path("/home/ubuntu/flashinfer-trace")
traceset = TraceSet.from_path(traceset_path)
selected_json_path = Path("/home/ubuntu/AccelOpt/experiments/flb_optimize/partial_selected_traces_triton.jsonl")
selected_traces = load_jsonl_file(Trace, selected_json_path)
output_table = []

for trace in selected_traces:
    # Create workloads
    trace_definition = traceset.definitions[trace.definition]
    workload_path = output_base_path / "workloads" / trace_definition.op_type / f"{get_workload_stem_name(trace)}.json"
    workload_trace = Trace(
        definition=trace.definition,
        workload=trace.workload,
    )
    save_json_file(workload_trace, workload_path)

    # Create solutions
    partial_solution = trace.solution
    solution = find_first(traceset.solutions[trace.definition], lambda x: x.name == partial_solution)
    assert solution is not None
    trace_definition = traceset.definitions[trace.definition]
    solution_path = output_base_path / "solutions" / trace_definition.op_type / trace.definition / f"{solution.name}.json"
    save_json_file(solution, solution_path)

    # Create definitions
    trace_definition = traceset.definitions[trace.definition]
    definition_path = output_base_path / "definitions" / trace_definition.op_type / f"{trace.definition}.json"
    save_json_file(trace_definition, definition_path)

    output_table.append({
        "definition_path": definition_path,
        "solution_path": solution_path,
        "workload_path": workload_path,
    })

output_df = pd.DataFrame(output_table)
output_df.to_csv(selected_json_path.with_suffix(".csv"), index=False)