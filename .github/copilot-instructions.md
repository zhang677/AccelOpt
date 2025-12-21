# AccelOpt Coding Agent Instructions

Use these instructions to be immediately productive in this repo. They capture the project’s architecture, key workflows, conventions, and integration points specific to AccelOpt.

## Big Picture
- AccelOpt is an agentic system that iteratively optimizes NKI (AWS Neuron Kernel Interface) kernels for Trainium, guided by planning prompts and performance profiling.
- Core library: `accelopt/` exposes wrappers to profile kernels and enforce correctness:
  - `NKIKernel` in `accelopt/kernel_wrapper.py` profiles NKI kernels against a NumPy reference module.
  - `FlashInferKernel` in `accelopt/flb_wrapper.py` integrates with `flashinfer-bench` traces/definitions/solutions.
- Data sources:
  - `NKIBench/` holds kernel programs and NumPy references; `summary.json` maps problems→cases→impls.
  - `experiments/flb_optimize/` mirrors FlashInfer-Bench artifacts (definitions, solutions, workloads, traces).
- Agents + prompts:
  - Planner produces optimization plans from profiles (`scripts/planner.py`, `prompts/planner_prompts/*`).
  - Executor generates code from plans and profiles candidates (`scripts/executor.py`, `prompts/executor_prompts/*`).

## How Things Flow
- Profile baseline kernels to establish latency + metrics, then propose code edits and re-profile to compute speedup.
- `scripts/executor.py` two-stage flow:
  - Stage 1: parallel LLM proposals using `user_prompt_template.txt` and baseline code/spec.
  - Stage 2: write temp kernel file, run `NKIKernel.profile(...)` with hard timeouts, collect `{latency, speedup, errors}`.
- Candidate selection merges best per-plan samples (`scripts/select_candidates.py`) and writes runnable files + CSV for batch profiling (`scripts/sequential_profile.py`).
- FlashInfer-Bench path: select traces → materialize definitions/solutions/workloads → profile baselines → compare.

## Core APIs and Required Patterns
- NumPy reference modules must define: `forward(*inputs)`, `get_inputs()`, `transform_to_nki_inputs(np_inputs)`, `transform_nki_outputs(nki_outputs, ref_output)`.
- `NKIKernel.profile(save_fields)` enforces:
  - Env: `NEURON_CC_FLAGS=--auto-cast=none`, `NEURON_RT_NUM_CORES=1`.
  - Correctness across multiple seeds using `accelopt.eval_numpy.check_precision_and_correctness`.
  - Precision rule: do not introduce `float16` in kernels; correctness check fails if `"float16"` appears in source.
  - Latency via `neuron-profile` CLI; repeats until `rel_diffs ≤ perf_tol` (retries capped).
- Saved metrics come from `prompts/profile_list.json` and are added to `res.metadata` when present.

## Conventions and IDs
- Case naming: `get_case_name(problem, values)` → `problem_KV` string; service names add `_ID{uuid}` via `init_service_name`.
- Plan/sample keys: `plan_{i}_{j}`; speedup computed as `baseline_latency / candidate_latency` when both exist.
- For mamba kernels a looser tolerance is used in profiling (`rel_tol=3e-5`).

## Critical Workflows (Commands)
- Install library:
  ```bash
  python setup.py install
  ```
- Profile one kernel (NKIBench):
  ```bash
  python tests/test_profile.py
  ```
- Build candidates and baseline profiles (NKIBench):
  ```bash
  export ACCELOPT_BASE_DIR="$(pwd)"
  python scripts/collect_candidates.py --output_candidates_path experiments/full_complete_local/candidates.csv \
    --output_profile_path experiments/full_complete_local/profile_results.csv \
    --save_fields_path prompts/profile_list.json --nc_id 0 --mode construct
  ```
- Run planner (produces per-service plans):
  ```bash
  python scripts/planner.py --output_path templates/complete_local/plans.json \
    --breadth 4 --exp_dir experiments/full_complete_local/results \
    --base_prompt_path prompts/planner_prompts/base_prompt.txt \
    --user_template_path prompts/planner_prompts/planner_prompt_template.txt \
    --profile_result_path experiments/full_complete_local/profile_results.csv \
    --model_config_path templates/complete_local/model_config.json \
    --displayed_profiles_path prompts/planner_prompts/displayed_profiles.json
  ```
- Run executor (generate implementations and profile):
  ```bash
  python scripts/executor.py --num_samples 4 --nc_id 0 \
    --problems_path experiments/full_complete_local/candidates.csv \
    --extractor_output_path templates/complete_local/plans.json \
    --exp_dir experiments/full_complete_local/results/$(date +%m-%d-%H-%M) \
    --base_prompt_path prompts/executor_prompts/base_prompt.txt \
    --user_template_path prompts/executor_prompts/user_prompt_template.txt \
    --save_fields_path prompts/profile_list.json \
    --model_config_path templates/complete_local/model_config.json \
    --output_path experiments/full_complete_local/results/executor_results.json \
    --exp_date $(date +%m-%d-%H-%M)
  ```
- Select top candidates and batch re-profile:
  ```bash
  python scripts/select_candidates.py --executor_results_path experiments/full_complete_local/results/executor_results.json \
    --output_base_path experiments/full_complete_local/results
  python scripts/sequential_profile.py --candidates_path experiments/full_complete_local/results/candidates.csv \
    --save_fields_path prompts/profile_list.json --output_path experiments/full_complete_local/results/profile_results.csv \
    --nc_id 0 --rel_tol 2e-5
  ```

## FlashInfer-Bench Workflow (examples)
- Select traces and materialize artifacts:
  ```bash
  python experiments/flb_optimize/select_traces.py
  python experiments/flb_optimize/create_files.py
  ```
- Profile baselines with checkpoints:
  ```bash
  cp -r /home/ubuntu/flashinfer-trace/blob experiments/flb_interface/checkpoints
  python experiments/flb_optimize/profile_baselines.py
  ```

## Integration Points
- AWS Neuron SDK: `neuronxcc.nki`, `neuron-profile` CLI must be available; set `NEURON_RT_VISIBLE_CORES` before profiling.
- LLM runtime: `openai-agents` via `AsyncOpenAI` with `model_config.json` (`url`, `api_key`, `model`). Reasoning settings enabled for Claude-like models.
- FlashInfer-Bench: `flashinfer_bench` python package and trace checkpoint directory (see `experiments/flb_interface/checkpoints`).

## Gotchas
- Do not mix advanced and basic indexing in NKI; follow prompt guidance in `prompts/executor_prompts/base_prompt.txt`.
- Avoid writing to the same tensor slice across parallel loops; use `nl.sequential_range` when necessary.
- Use only APIs already present in the baseline kernel; do not invent new NKI APIs.

References: `accelopt/kernel_wrapper.py`, `accelopt/eval_numpy.py`, `scripts/*`, `prompts/*`, `NKIBench/summary.json`, `experiments/*`, top-level `README.md`.