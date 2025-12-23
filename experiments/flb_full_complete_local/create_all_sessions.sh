#!/bin/bash
exp_date_base="12-21-17-05"
exp_dir="../checkpoints/$exp_date_base"
scripts=(
    "run_single_loop_gqa_paged_decode_h32_kv4_d128_ps1_7a7fc28_gpt-5_triton_f88811.sh"
    "run_single_loop_gqa_paged_prefill_causal_h32_kv8_d128_ps1_25f2945_gemini-2.5-pro_triton_3j61np.sh"
    "run_single_loop_gqa_ragged_prefill_causal_h32_kv4_d128_007ddab_claude-opus-4-1_triton_28277f.sh"
    "run_single_loop_mla_paged_decode_h16_ckv512_kpe64_ps1_939f995_gpt-o3_triton_4c17a1.sh"
    "run_single_loop_mla_paged_prefill_causal_h16_ckv512_kpe64_ps1_733a7bb_gpt-5_triton_88089a.sh"
    "run_single_loop_moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048_5e8dc11_gpt-o3_triton_c1adb5.sh"
    "run_single_loop_rmsnorm_h128_f2872f8_gpt-o3_triton_35b90e.sh"
)

for i in "${!scripts[@]}"; do
  sess="sample-$exp_date_base-$((i+1))"
  # Use bash -lc so 'source' works (it's a bash builtin) and PATH/profile apply
  tmux new-session -d -s "$sess" \
    "bash -lc 'cd \"$exp_dir\" \
      && export ACCELOPT_BASE_DIR=\"${ACCELOPT_BASE_DIR}\" \
      && bash \"${scripts[$i]}\"; exec bash'"
done