# Workload criteria: The first workload whose baseline latency is larger than 20ms. If not found, select the one with largest baseline latency
# Input: TraceSet root
# Output: A map from workload per definition to baseline latency
from flashinfer_bench import TraceSet, EvaluationStatus, SupportedLanguages

def find_first(list, predicate):
    for item in list:
        if predicate(item):
            return item
    return None

traceset_path = "/home/ubuntu/flashinfer-trace"
traceset = TraceSet.from_path(traceset_path)
all_traces = traceset.traces
trace_map = {}
for definition_name in all_traces.keys():
    trace_map[definition_name] = {}
    for trace in all_traces[definition_name]:
        if trace.evaluation.performance is not None and trace.evaluation.performance.latency_ms is not None and trace_map[definition_name].get(trace.workload.uuid) is None:
            trace_map[definition_name][trace.workload.uuid] = trace.evaluation.performance

selected_workloads = {}
for definition_name, workload_map in trace_map.items():
# The first workload whose baseline latency is larger than 20ms. If not found, select the one with largest baseline latency
    # Sort by baseline latency
    sorted_workloads = sorted(workload_map.items(), key=lambda x: x[1].latency_ms)
    for workload, evaluation in sorted_workloads:
        if evaluation.latency_ms > 10:
            selected_workload_uuid = workload
            selected_evaluation = evaluation
            break
    else:
        selected_workload_uuid = sorted_workloads[-1][0]
        selected_evaluation = sorted_workloads[-1][1]
    selected_workload = find_first(traceset.workloads[definition_name], lambda x: x.workload.uuid == selected_workload_uuid)
    selected_workloads[definition_name] = {
        "workload": selected_workload,
        "evaluation": selected_evaluation
    }

    print(definition_name, selected_workload, selected_evaluation.latency_ms)

# For each item in selected_workloads, find the triton solution that is PASSED and has the largest speedup
selected_workloads_triton = []
for definition_name in all_traces.keys():
    selected_workload = selected_workloads[definition_name]["workload"]
    candidate_traces = []
    for trace in [trace for trace in all_traces[definition_name] if trace.workload.uuid == selected_workload.workload.uuid]:
        if trace.evaluation.status == EvaluationStatus.PASSED:
            solution = find_first(traceset.solutions[definition_name], lambda x: x.name == trace.solution)
            assert solution is not None
            if solution.spec.language == SupportedLanguages.TRITON:
                candidate_traces.append(trace)
    if len(candidate_traces) > 0:
        selected_solution = max(candidate_traces, key=lambda x: x.evaluation.performance.speedup_factor)
        selected_workloads_triton.append(selected_solution)

with open("selected_traces_triton.jsonl", "w") as f:
    for solution in selected_workloads_triton:
        f.write(solution.model_dump_json() + "\n")


custom_prefix_list = [
    "gemm_n",
    "gqa_paged_decode_h",
    "gqa_paged_prefill_causal_h",
    "gqa_ragged_prefill_causal_h",
    "mla_paged_decode_h",
    "mla_paged_prefill_causal_h",
    "moe",
    "fused_add_rmsnorm_h",
    "rmsnorm_h"
]

# Group by prefix of definitions and select the one with largest latency_ms
grouped_selected_workloads = {}
for trace in selected_workloads_triton:
    definition_name = trace.definition
    for prefix in custom_prefix_list:
        if definition_name.startswith(prefix):
            group_key = prefix
            break
    if group_key not in grouped_selected_workloads:
        grouped_selected_workloads[group_key] = [trace]
    else:
        grouped_selected_workloads[group_key].append(trace)
final_selected_workloads = []
for group_key, traces in grouped_selected_workloads.items():
    selected_trace = max(traces, key=lambda x: x.evaluation.performance.latency_ms)
    final_selected_workloads.append(selected_trace)
with open("partial_selected_traces_triton.jsonl", "w") as f:
    for solution in final_selected_workloads:
        f.write(solution.model_dump_json() + "\n")
