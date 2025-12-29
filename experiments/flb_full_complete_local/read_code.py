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
# solution_path = "/home/ubuntu/AccelOpt/experiments/flb_optimize/solutions/rmsnorm/fused_add_rmsnorm_h7168/gemini-2.5-pro_triton_05pwmx.json"
# solution_path = "/home/ubuntu/AccelOpt/experiments/flb_interface/checkpoints/solutions/rmsnorm/fused_add_rmsnorm_h7168/194718f09cd644fa85de20979839da4f.json"

# gqa_ragged_prefill
# solution_path = "/home/ubuntu/AccelOpt/experiments/flb_interface/checkpoints/solutions/gqa_ragged/gqa_ragged_prefill_causal_h32_kv4_d128/40724a7193cf42eda2d03426dfaf4b50.json"
# solution_path = "/home/ubuntu/AccelOpt/experiments/flb_optimize/solutions/gqa_ragged/gqa_ragged_prefill_causal_h32_kv4_d128/claude-opus-4-1_triton_28277f.json"

# gqa_paged_decode
# solution_path = "/home/ubuntu/AccelOpt/experiments/flb_interface/checkpoints/solutions/gqa_paged/gqa_paged_decode_h32_kv4_d128_ps1/54fc7efa3d6045c3a2acbb497d086e73.json"
# solution_path = "/home/ubuntu/AccelOpt/experiments/flb_optimize/solutions/gqa_paged/gqa_paged_decode_h32_kv4_d128_ps1/gpt-5_triton_f88811.json"

# moe
# solution_path = "/home/ubuntu/AccelOpt-exps/checkpoints/12-26-10-18/solutions/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048/gpt-o3_triton_c1adb5.json"
# solution_path = "/home/ubuntu/AccelOpt/experiments/flb_interface/checkpoints/solutions/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048/7222798c4446466a8d7774756ce8e7b4.json"
solution_path = "/home/ubuntu/AccelOpt/experiments/flb_interface/checkpoints/solutions/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048/6869031cb17b4a8a9b73979e932951b0.json"
solution = load_json_file(Solution, solution_path)
with open("temp_opt.py", "w") as f:
    f.write(solution.sources[0].content)