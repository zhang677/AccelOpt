from flashinfer_bench import Solution
from flashinfer_bench.data import load_json_file

# GEMM
# solution_path = "/home/ubuntu/AccelOpt/experiments/flb_optimize/solutions/gemm/gemm_n28672_k4096/gpt-o3_triton_4c9c32.json"
# solution_path = "/home/ubuntu/AccelOpt/experiments/flb_optimize/solutions/gqa_paged/gqa_paged_decode_h32_kv4_d128_ps1/gpt-5_triton_f88811.json"
# solution_path = "/home/ubuntu/AccelOpt/experiments/flb_optimize/solutions/gqa_paged/gqa_paged_prefill_causal_h32_kv8_d128_ps1/gemini-2.5-pro_triton_3j61np.json"
# solution_path = "/home/ubuntu/AccelOpt/experiments/flb_interface/checkpoints/solutions/gemm/gemm_n28672_k4096/9af6ff2bc04f4ab899d9db16bd9544e0.json"
# solution_path = "/home/ubuntu/AccelOpt/experiments/flb_interface/checkpoints/solutions/gemm/gemm_n28672_k4096/d5e6cac005ac428f9b70bf5192f14490.json"
# solution_path = "/home/ubuntu/AccelOpt/experiments/flb_interface/checkpoints/solutions/gemm/gemm_n28672_k4096/e5926f58916a43daa59f6f5cd3d84fa6.json"

# fused_add_rmsnorm
solution_path = "/home/ubuntu/AccelOpt/experiments/flb_optimize/solutions/rmsnorm/fused_add_rmsnorm_h7168/gemini-2.5-pro_triton_05pwmx.json"
# solution_path = "/home/ubuntu/AccelOpt/experiments/flb_interface/checkpoints/solutions/rmsnorm/fused_add_rmsnorm_h7168/194718f09cd644fa85de20979839da4f.json"
solution = load_json_file(Solution, solution_path)
with open("temp.py", "w") as f:
    f.write(solution.sources[0].content)