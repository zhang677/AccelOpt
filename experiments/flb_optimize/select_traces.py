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
        if trace.evaluation.performance is not None and trace.evaluation.performance.reference_latency_ms is not None and trace_map[definition_name].get(trace.workload.uuid) is None:
            trace_map[definition_name][trace.workload.uuid] = trace.evaluation.performance

selected_workloads = {}
for definition_name, workload_map in trace_map.items():
# The first workload whose baseline latency is larger than 20ms. If not found, select the one with largest baseline latency
    # Sort by baseline latency
    sorted_workloads = sorted(workload_map.items(), key=lambda x: x[1].reference_latency_ms)
    for workload, evaluation in sorted_workloads:
        if evaluation.reference_latency_ms > 20:
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

    print(definition_name, selected_workload, selected_evaluation.reference_latency_ms)

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

with open("selected_workloads_triton.jsonl", "w") as f:
    for solution in selected_workloads_triton:
        f.write(solution.model_dump_json() + "\n")

