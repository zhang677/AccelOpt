import json, uuid, time, tempfile
import os
import traceback
import numpy as np
from .eval_numpy import load_module_from_path, check_precision_and_correctness
from pydantic import BaseModel, Field
import neuronxcc.nki as nki
from flashinfer_bench import TraceSet, Solution, SupportedLanguages, Definition, SourceFile, BuildSpec, Trace, Evaluation, Benchmark, BenchmarkConfig, EvaluationStatus
from flashinfer_bench.data import save_json_file, load_json_file
from pathlib import Path

def get_latency(nki_kernel_fn, nki_inputs, artifact_dir):
    kernel_id = uuid.uuid4()
    neff_path = os.path.join(artifact_dir, f"neff_{kernel_id}.neff")
    ntff_path = os.path.join(artifact_dir, f"ntff_{kernel_id}.ntff")
    nki.baremetal(
        nki_kernel_fn,
        save_neff_name=neff_path,
        save_trace_name=ntff_path,
        additional_compile_opt="--disable-dge --logical-nc-config=1"
    )(*nki_inputs)
    summary_profile_path = os.path.join(artifact_dir, f"profile_{kernel_id}.json")

    summary_profile_cmd = f"neuron-profile view --output-format summary-json -n {neff_path} -s {ntff_path} > {summary_profile_path}"
    os.system(summary_profile_cmd)
    summary = json.load(open(summary_profile_path, 'r'))
    latency_ms = summary[next(iter(summary))]["total_time"] * 1e3
    return latency_ms

def benchmark_latency(warmpup_iterations, benchmark_iterations, nki_kernel_fn, nki_inputs, artifact_dir):
    for _ in range(warmpup_iterations):
        nki.baremetal(
            nki_kernel_fn,
            additional_compile_opt="--disable-dge --logical-nc-config=1"
        )(*nki_inputs)
    latency_ms_list = []
    for _ in range(benchmark_iterations):
        latency_ms = get_latency(nki_kernel_fn, nki_inputs, artifact_dir)
        latency_ms_list.append(latency_ms)
    runtime_stats = {
        "mean_ms": np.mean(latency_ms_list),
        "min_ms": np.min(latency_ms_list),
        "max_ms": np.max(latency_ms_list),
        "rel_diffs": (np.max(latency_ms_list) - np.min(latency_ms_list)) / np.min(latency_ms_list)
    }
    return runtime_stats

class KernelProperties(BaseModel):
    """
    Single Kernel Execution
    """
    compiled: bool = False
    correct: bool = False
    runnable: bool = False
    metadata: dict = Field(default_factory=dict)

class NKIKernel:
    def __init__(self, program_path: str, base_numpy_path: str):
        self.program_path = program_path
        self.base_numpy_path = base_numpy_path
        self.res = KernelProperties()
        self.rel_tol = 2e-5
        self.perf_tol = 0.01

    def profile(self, save_fields: list[str] = []):
        os.environ["NEURON_CC_FLAGS"] = "--auto-cast=none"
        os.environ['NEURON_RT_NUM_CORES']= '1'
        np.random.seed(42)
        task_module = load_module_from_path(self.base_numpy_path)
        task_fn = task_module.forward
        task_np_input_fn = task_module.get_inputs
        task_np_inputs = task_np_input_fn()
        task_nki_output_fn = task_module.transform_nki_outputs
        self.res = KernelProperties()
        new_profile_name = f"nki_{uuid.uuid4()}"

        with tempfile.TemporaryDirectory(dir="/tmp", prefix=f"{new_profile_name}_") as artifact_dir:
            neff_path = os.path.join(artifact_dir, f"kernel_file.neff")
            ntff_path = os.path.join(artifact_dir, f"kernel_profile.ntff")
            try:
                nki_kernel_module = load_module_from_path(self.program_path)
                if hasattr(nki_kernel_module, "kernel"):
                    nki_kernel_fn = nki_kernel_module.kernel
                elif hasattr(nki_kernel_module, "optimized_kernel"):
                    nki_kernel_fn = nki_kernel_module.optimized_kernel
                else:
                    raise ValueError(f"No kernel function found in {self.program_path}")
                # Get the transform_to_nki_inputs function
                if hasattr(task_module, "transform_to_nki_inputs"):
                    task_nki_input_fn = task_module.transform_to_nki_inputs
                else:
                    raise ValueError(f"No transform_to_nki_inputs function found in {self.program_path} or {self.base_numpy_path}")
                nki_inputs = task_nki_input_fn(task_np_inputs)
                
                output_nki = nki.baremetal(
                    nki_kernel_fn,
                    save_neff_name=neff_path,
                    save_trace_name=ntff_path,
                    additional_compile_opt="--disable-dge --logical-nc-config=1"
                )(*nki_inputs)
                self.res.compiled = True
                self.res.runnable = True
                    
            except Exception as e:
                print(f"Compilation failure. Error: {e}")
                self.res.metadata["compilation_error"] = str(e)
                self.res.metadata["compilation_traceback"] = traceback.format_exc()
                return self.res

            try:
                for rnd_seed in [0, 21, 42, 63, 84]:
                    np.random.seed(rnd_seed)
                    task_np_inputs = task_np_input_fn()
                    nki_inputs = task_nki_input_fn(task_np_inputs)
                    output_task = task_fn(*task_np_inputs)
                    output_nki_raw = nki.baremetal(
                        nki_kernel_fn,
                        additional_compile_opt="--disable-dge --logical-nc-config=1"
                    )(*nki_inputs)
                    output_nki = task_nki_output_fn(output_nki_raw, output_task)
                    check_precision_and_correctness(self.program_path, output_nki, output_task, self.res, self.rel_tol)
                    if not self.res.correct:
                        break
            except Exception as e:
                print(f"Correct checking failure. Error: {e}")
                self.res.metadata["correctness_error"] = str(e)
                return self.res
            
            if not self.res.correct:
                return self.res

            try:
                runtime_stats = benchmark_latency(2, 10, nki_kernel_fn, nki_inputs, artifact_dir)
                rel_diff = runtime_stats["rel_diffs"]
                rel_diff_list = [rel_diff]
                runtime_stats_list = [runtime_stats]
                while rel_diff > self.perf_tol:
                    print(f"Retry: {self.program_path } at {len(rel_diff_list)}; rel_diffs: {rel_diff_list}")
                    time.sleep(1)
                    
                    rel_diff_list.append(rel_diff)
                    runtime_stats_list.append(runtime_stats)
                    if len(rel_diff_list) > 2: # Just retry twice. In paper, we did 10 times.
                        break
                runtime_stats = runtime_stats_list[np.argmin(rel_diff_list)]
                self.res.metadata["latency"] = runtime_stats["mean_ms"]
                self.res.metadata["min_ms"] = runtime_stats["min_ms"]
                self.res.metadata["max_ms"] = runtime_stats["max_ms"]
                self.res.metadata["rel_diffs"] = runtime_stats["rel_diffs"]

                summary_profile_path = os.path.join(artifact_dir, f"{new_profile_name}_summary_profile.json")
                summary_profile_cmd = f"neuron-profile view --output-format summary-json -n {neff_path} -s {ntff_path} > {summary_profile_path}"
                os.system(summary_profile_cmd)
                summary = json.load(open(summary_profile_path, 'r'))
                profile_result = summary[next(iter(summary))]
                for field in save_fields:
                    if field in profile_result.keys():
                        self.res.metadata[field] = profile_result[field]
            except Exception as e:
                print(f"Benchmarking failure. Error: {e}")
                self.res.metadata["benchmarking_error"] = traceback.format_exc()
                return self.res
            return self.res

class FlashInferKernel:
    def __init__(self, traceset_path: str, definition_path: str):
        self.traceset = TraceSet.from_path(traceset_path)
        self.definition = load_json_file(Definition, definition_path)

    def profile(self, solution_path, workload_path: str, **kwargs) -> Evaluation:
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