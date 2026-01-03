# AccelOpt: service_name,task,kernel,problem,values,case_name
from flashinfer_bench import TraceSet, Solution, SupportedLanguages, Definition, SourceFile, BuildSpec, Trace, Evaluation, Benchmark, BenchmarkConfig, EvaluationStatus
from flashinfer_bench.data import save_json_file, load_json_file
from pathlib import Path
from pydantic import BaseModel, Field
from accelopt.flb_wrapper import FlashInferKernel
import uuid
import logging

logging.basicConfig(level=logging.INFO)

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
    definition_path = "/home/ubuntu/AccelOpt/experiments/flb_optimize/definitions/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.json"
    workload_path = "/home/ubuntu/AccelOpt/experiments/flb_optimize/workloads/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048_5e8dc11.json"
    baseline_path = "/home/ubuntu/AccelOpt/experiments/flb_optimize/solutions/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048/gpt-o3_triton_c1adb5.json"
    # baseline_path = "/home/ubuntu/AccelOpt/experiments/flb_optimize/solutions/mla_paged/mla_paged_decode_h16_ckv512_kpe64_ps1/gpt-o3_triton_4c17a1.json"
    # definition_path = "/home/ubuntu/AccelOpt/experiments/flb_optimize/definitions/mla_paged/mla_paged_decode_h16_ckv512_kpe64_ps1.json"
    # workload_path = "/home/ubuntu/AccelOpt/experiments/flb_optimize/workloads/mla_paged/mla_paged_decode_h16_ckv512_kpe64_ps1_939f995.json"
    # definition_path = "/home/ubuntu/AccelOpt/experiments/flb_optimize/definitions/mla_paged/mla_paged_prefill_causal_h16_ckv512_kpe64_ps1.json"
    # baseline_path = "/home/ubuntu/AccelOpt/experiments/flb_optimize/solutions/mla_paged/mla_paged_prefill_causal_h16_ckv512_kpe64_ps1/gpt-5_triton_88089a.json"
    # workload_path = "/home/ubuntu/AccelOpt/experiments/flb_optimize/workloads/mla_paged/mla_paged_prefill_causal_h16_ckv512_kpe64_ps1_733a7bb.json"
    baseline_kernel = FlashInferKernel(checkpoint_path, definition_path)
    baseline_trace, baseline_res = baseline_kernel.profile(
        baseline_path,
        workload_path=workload_path,
        timeout_seconds=300,
        profile_baseline=True,
        use_isolated_runner=True,
        destination_passing_style=True
    )
    print(baseline_trace.evaluation.performance.reference_latency_ms)
    save_json_file(baseline_trace, "temp.json")
    
    
    # Register a solution to FlashInfer-Trace
    # mock_kernel_path = baseline_path
    # mock_solution = load_json_file(Solution, mock_kernel_path)
    # kernel_code = mock_solution.sources[0].content
    # kernel = FlashInferKernel(checkpoint_path, definition_path)
    # solution_path = kernel.create_and_save_solution(
    #     kernel_code,
    #     author="AccelOpt",
    #     language="triton",
    #     target_gpu="H100",
    #     name=f"optimized_{uuid.uuid4()}",
    #     description=f"Optimized kernel"
    # )
    # res = kernel.profile(
    #     solution_path, 
    #     workload_path=workload_path,
    #     timeout_seconds=300,
    #     profile_baseline=True,
    #     use_isolated_runner=True
    # )

    # print(f"Solution saved to: {solution_path}")

    # os.remove(solution_path)
    # print(f"Solution removed: {solution_path}")

