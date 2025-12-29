# AccelOpt: service_name,task,kernel,problem,values,case_name
from flashinfer_bench import TraceSet, Solution, SupportedLanguages, Definition, SourceFile, BuildSpec, Trace, Evaluation, Benchmark, BenchmarkConfig, EvaluationStatus
from flashinfer_bench.data import save_json_file, load_json_file
from pathlib import Path
from pydantic import BaseModel, Field
import json
from accelopt.flb_wrapper import FlashInferKernel
import uuid
# Profile a solution https://github.com/flashinfer-ai/flashinfer-bench/blob/main/examples/kernel_generator/kernel_generator.py#L445
# Does FlashInfer-Bench prefer string or file path?
# (traceset, definition, [solution], selected_workload)
# self.traceset
# Path to definition
# code of solution
# Path to workload

class KernelProperties(BaseModel):
    """
    Single Kernel Execution
    """
    compiled: bool = False
    correct: bool = False
    runnable: bool = False
    metadata: dict = Field(default_factory=dict)

class FlashInferKernelLocal:
    def __init__(self, traceset_path: str, definition_path: str):
        self.traceset = TraceSet.from_path(traceset_path)
        self.definition = load_json_file(Definition, definition_path)

    def profile(self, solution_path, workload_path: str, **kwargs) -> Evaluation:
        res = KernelProperties()
        definition = self.definition
        solution = load_json_file(Solution, solution_path)
        selected_workload = load_json_file(Trace, workload_path)
        temp_traceset = TraceSet(
            root=self.traceset.root,
            definitions={definition.name: definition},
            solutions={definition.name: [solution]},
            workloads={definition.name: [selected_workload, selected_workload]},
            traces={definition.name: []},
        )
        cfg = BenchmarkConfig()
        cfg.profile_baseline = False
        cfg.timeout_seconds = kwargs.get("timeout_seconds", 300)
        benchmark = Benchmark(temp_traceset, cfg)
        result_traceset = benchmark.run_all()

        traces = result_traceset.traces.get(definition.name, [])

        trace_map = {trace.solution: trace for trace in traces}
        evaluation = trace_map.get(solution.name).evaluation
        if evaluation.status == EvaluationStatus.PASSED:
            res.compiled = True
            res.runnable = True
            res.correct = True
            res.metadata = {"latency": evaluation.performance.latency_ms}
        elif evaluation.status in [EvaluationStatus.INCORRECT_SHAPE, EvaluationStatus.INCORRECT_NUMERICAL, EvaluationStatus.INCORRECT_DTYPE]:
            res.compiled = True
            res.runnable = True
            res.correct = False
            res.metadata = {"correctness_error": evaluation.log}
        elif evaluation.status == EvaluationStatus.RUNTIME_ERROR:
            res.compiled = True
            res.runnable = False
            res.correct = False
            res.metadata = {"runtime_error": evaluation.log}
        elif evaluation.status in [EvaluationStatus.COMPILATION_ERROR, EvaluationStatus.TIMEOUT]:
            res.compiled = False
            res.runnable = False
            res.correct = False
            res.metadata = {"compilation_error": evaluation.log}
        else:
            raise ValueError(f"Unsupported evaluation status: {evaluation.status}")
        return res

    def save_solution(self, code, **kwargs) -> Path:
        definition = self.definition
        solutions_dir = (
            self.traceset.root / "solutions" / definition.op_type / definition.name
        )
        solutions_dir.mkdir(parents=True, exist_ok=True)

        # Create filename using solution name
        solution = self._create_solution_from_code(code, definition, **kwargs)
        solution_filename = f"{solution.name}.json"
        solution_path = solutions_dir / solution_filename

        save_json_file(solution, solution_path)
        return solution_path

    def _get_supported_language(self, language: str) -> SupportedLanguages:
        language_map = {
            "triton": SupportedLanguages.TRITON,
        }
        if language.lower() in language_map:
            return language_map[language.lower()]
        else:
            raise ValueError(f"Unsupported language: {language}")

    def _create_solution_from_code(
        self, code, definition: Definition, **kwargs
    ) -> Solution:
        solution_name = kwargs.get("name", "default_solution")
        solution_description = kwargs.get("description", "default_description")
        language = self._get_supported_language(kwargs.get("language", "triton"))

        if language == SupportedLanguages.CUDA and isinstance(code, dict):
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
            author=kwargs.get("author", "AccelOpt"),
            spec=BuildSpec(
                language=language,
                target_hardware=[kwargs.get("target_gpu", "H100")],
                entry_point=entry_point,
            ),
            sources=sources,
            description=solution_description,
        )
        return solution


if __name__ == "__main__":
    # Check all definitions
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
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
    

    checkpoint_path = "/home/ubuntu/AccelOpt/experiments/flb_interface/checkpoints"
    
    # Profile baseline kernel
    definition_path = "/home/ubuntu/AccelOpt/experiments/flb_optimize/definitions/gemm/gemm_n28672_k4096.json"
    workload_path = "/home/ubuntu/AccelOpt/experiments/flb_optimize/workloads/gemm/gemm_n28672_k4096_32cd269.json"
    baseline_path = "/home/ubuntu/AccelOpt/experiments/flb_optimize/solutions/gemm/gemm_n28672_k4096/gpt-o3_triton_4c9c32.json"
    baseline_kernel = FlashInferKernel(checkpoint_path, definition_path)
    baseline_res = baseline_kernel.profile(
        baseline_path,
        workload_path=workload_path,
        timeout_seconds=300,
        profile_baseline=True,
        use_isolated_runner=True
    )
    print(baseline_res)
    
    
    # Register a solution to FlashInfer-Trace
    mock_kernel_path = baseline_path
    mock_solution = load_json_file(Solution, mock_kernel_path)
    kernel_code = mock_solution.sources[0].content
    kernel = FlashInferKernel(checkpoint_path, definition_path)
    solution_path = kernel.create_and_save_solution(
        kernel_code,
        author="AccelOpt",
        language="triton",
        target_gpu="H100",
        name=f"optimized_{uuid.uuid4()}",
        description=f"Optimized kernel"
    )
    res = kernel.profile(
        solution_path, 
        workload_path=workload_path,
        timeout_seconds=300,
        profile_baseline=True,
        use_isolated_runner=True
    )
    print(res)

    print(f"Solution saved to: {solution_path}")

    os.remove(solution_path)
    print(f"Solution removed: {solution_path}")

