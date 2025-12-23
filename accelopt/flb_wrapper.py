from .eval_numpy import KernelProperties
from pathlib import Path
from flashinfer_bench import TraceSet, Solution, SupportedLanguages, Definition, SourceFile, BuildSpec, Trace, Evaluation, Benchmark, BenchmarkConfig, EvaluationStatus
from flashinfer_bench.data import save_json_file, load_json_file



def _get_supported_language(language: str) -> SupportedLanguages:
    language_map = {
        "triton": SupportedLanguages.TRITON,
    }
    if language.lower() in language_map:
        return language_map[language.lower()]
    else:
        raise ValueError(f"Unsupported language: {language}")


def _create_solution_from_code(
    code, definition: Definition, **kwargs
) -> Solution:
    solution_name = kwargs.get("name", "default_solution")
    solution_description = kwargs.get("description", "default_description")
    language = _get_supported_language(kwargs.get("language", "triton"))

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


def create_and_save_solution(traceset: TraceSet, definition: Definition, code, **kwargs) -> str:
    solutions_dir = (
        traceset.root / "solutions" / definition.op_type / definition.name
    )
    solutions_dir.mkdir(parents=True, exist_ok=True)

    # Create filename using solution name
    solution = _create_solution_from_code(code, definition, **kwargs)
    solution_filename = f"{solution.name}.json"
    solution_path = solutions_dir / solution_filename

    save_json_file(solution, solution_path)
    return str(solution_path)

class FlashInferKernel:
    def __init__(self, traceset_root: str, definition_path: str):
        self.traceset = TraceSet.from_path(traceset_root)
        self.definition = load_json_file(Definition, definition_path)

    def profile(self, solution_path, workload_path: str, **kwargs) -> tuple[Trace, KernelProperties]:
        # Only support single workload for now
        res = KernelProperties()
        definition = self.definition
        solution = load_json_file(Solution, solution_path)
        selected_workload = load_json_file(Trace, workload_path)
        temp_traceset = TraceSet(
            root=self.traceset.root,
            definitions={definition.name: definition},
            solutions={definition.name: [solution]},
            workloads={definition.name: [selected_workload]},
            traces={definition.name: []},
        )
        cfg = BenchmarkConfig()
        cfg.profile_baseline = kwargs.get("profile_baseline", False)
        cfg.use_isolated_runner = kwargs.get("use_isolated_runner", True) # The FlashInferKernel abstraction assumes a isolated runner
        cfg.timeout_seconds = kwargs.get("timeout_seconds", 300)
        benchmark = Benchmark(temp_traceset, cfg)
        result_traceset = benchmark.run_all()
        
        traces = result_traceset.traces.get(definition.name, [])

        trace_map = {trace.solution: trace for trace in traces}
        single_trace = trace_map.get(solution.name)
        evaluation = single_trace.evaluation

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
            res.metadata = {"run_error": evaluation.log}
        elif evaluation.status in [EvaluationStatus.COMPILE_ERROR, EvaluationStatus.TIMEOUT]:
            res.compiled = False
            res.runnable = False
            res.correct = False
            res.metadata = {"compilation_error": evaluation.log}
        else:
            raise ValueError(f"Unsupported evaluation status: {evaluation.status}")
        return single_trace, res

    def create_and_save_solution(self, code, **kwargs) -> str:
        return create_and_save_solution(self.traceset, self.definition, code, **kwargs)


# https://github.com/flashinfer-ai/flashinfer-bench/blob/93bf860d9730dec3fc990ce38ce01814ffea4118/examples/kernel_generator/kernel_generator_prompts.py#L8
def format_definition(definition: Definition) -> str:
    axes_str = "\nAxes:\n"
    for name, axis in definition.axes.items():
        if hasattr(axis, "value"):
            axes_str += f"  {name}: constant = {axis.value}"
        else:
            axes_str += f"  {name}: variable"
        if axis.description:
            axes_str += f" ({axis.description})"
        axes_str += "\n"

    # Format inputs
    inputs_str = "\nInputs:\n"
    for name, spec in definition.inputs.items():
        shape_str = "scalar" if spec.shape is None else f"[{', '.join(spec.shape)}]"
        inputs_str += f"  {name}: {shape_str} ({spec.dtype})"
        if spec.description:
            inputs_str += f" - {spec.description}"
        inputs_str += "\n"

    outputs_str = "\nOutputs:\n"
    for name, spec in definition.outputs.items():
        shape_str = "scalar" if spec.shape is None else f"[{', '.join(spec.shape)}]"
        outputs_str += f"  {name}: {shape_str} ({spec.dtype})"
        if spec.description:
            outputs_str += f" - {spec.description}"
        outputs_str += "\n"

    constraints_str = ""
    if definition.constraints:
        constraints_str = "\nConstraints:\n"
        for constraint in definition.constraints:
            constraints_str += f"  - {constraint}\n"

    return f"""Name: {definition.name}
Type: {definition.op_type}
{axes_str}{inputs_str}{outputs_str}{constraints_str}

Reference Implementation:
{definition.reference}"""

def get_unique_trace_name(solution: Solution, workload: Trace) -> str:
    return f"{workload.definition}_{workload.workload.uuid[:7]}_{solution.name}"

def get_workload_stem_name(workload: Trace) -> str:
    return f"{workload.definition}_{workload.workload.uuid[:7]}"