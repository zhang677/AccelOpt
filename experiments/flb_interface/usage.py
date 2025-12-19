# AccelOpt: service_name,task,kernel,problem,values,case_name
from flashinfer_bench import TraceSet, Solution, SupportedLanguages, Definition, SourceFile, BuildSpec, Trace, Evaluation, Benchmark, BenchmarkConfig
import json
# Profile a solution https://github.com/flashinfer-ai/flashinfer-bench/blob/main/examples/kernel_generator/kernel_generator.py#L445
# Does FlashInfer-Bench prefer string or file path?
# (traceset, definition, [solution], selected_workload)
# self.traceset
# Path to definition
# code of solution
# Path to workload

class FlashInferKernel:
    def __init__(self, traceset_path: str, author: str, language: str, target_gpu: str):
        self.traceset = TraceSet.from_path(traceset_path)
        self.author = author
        self.language = self._get_supported_language(language)
        self.target_gpu = target_gpu

    def profile(self, code, definition_path: str, workload_path: str, solution_metadata) -> Evaluation:
        definition = Definition.model_validate_json(open(definition_path, "r").read())
        solution = self._create_solution_from_code(code, definition, solution_metadata)
        selected_workload = Trace.model_validate_json(open(workload_path, "r").read())
        temp_traceset = TraceSet(
            root=self.traceset.root,
            definitions={definition.name: definition},
            solutions={definition.name: [solution]},
            workloads={definition.name: [selected_workload]},
            traces={definition.name: []},
        )
        benchmark = Benchmark(temp_traceset, BenchmarkConfig())
        result_traceset = benchmark.run_all()

        traces = result_traceset.traces.get(definition.name, [])

        trace_map = {trace.solution: trace for trace in traces}
        evaluation = trace_map.get(solution.name)
        return evaluation

    def _get_supported_language(self, language: str) -> SupportedLanguages:
        language_map = {
            "triton": SupportedLanguages.TRITON,
        }
        if language.lower() in language_map:
            return language_map[language.lower()]
        else:
            raise ValueError(f"Unsupported language: {language}")

    def _create_solution_from_code(
        self, code, definition: Definition, solution_metadata: dict
    ) -> Solution:
        solution_name = solution_metadata["name"]
        solution_description = solution_metadata["description"]

        if self.language == SupportedLanguages.CUDA and isinstance(code, dict):
            sources = []
            for filename, content in code.items():
                sources.append(SourceFile(path=filename, content=content))

            entry_point = "main.cpp::run"
        else:
            if isinstance(code, dict):
                code = next(iter(code.values()))

            sources = [SourceFile(path="main.py", content=code)]
            entry_point = "main.py::run"

        solution = Solution(
            name=solution_name,
            definition=definition.name,
            author=self.author,
            spec=BuildSpec(
                language=self.language,
                target_hardware=[self.target_gpu],
                entry_point=entry_point,
            ),
            sources=sources,
            description=solution_description,
        )
        return solution


if __name__ == "__main__":
    # Check all definitions
    traceset_path = "/home/ubuntu/flashinfer-trace"
    print(f"Loading TraceSet from: {traceset_path}")
    traceset = TraceSet.from_path(traceset_path)

    all_definitions = traceset.definitions
    all_definitions_keys = list(all_definitions.keys()) # case_name
    # print(all_definitions_keys)
    case_name = 'gemm_n128_k2048'
    single_definition = all_definitions[case_name]
    problem = single_definition.op_type # problem
    print(problem)
    symbolic_values = single_definition.axes # symbolic value
    print(symbolic_values)

    # Concrete values are needed to calculate the theoretical peak
    all_workloads = traceset.workloads
    single_workload = all_workloads[case_name][-3] # concrete value
    print(single_workload)


    # Retrieve a solution
    all_solutions = traceset.solutions
    case_solutions = all_solutions[case_name]
    case_single_solution = case_solutions[-2]
    assert case_single_solution.spec.language == 'triton'
    kernel_code = case_single_solution.sources[0].content # kernel code

    task_code = single_definition.reference # task code
    print(task_code)

    kernel = FlashInferKernel(traceset_path, "AccelOpt", "triton", "H100")
    evaluation = kernel.profile(
        kernel_code, 
        "/home/ubuntu/flashinfer-trace/definitions/gemm/gemm_n128_k2048.json",
        "/home/ubuntu/AccelOpt/experiments/flb_interface/example_workload.jsonl",
        {
            "name": f"{case_name}_optimized",
            "description": f"Optimized kernel for {case_name}"
        }
    )
    print(evaluation)

