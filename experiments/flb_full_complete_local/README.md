export ACCELOPT_BASE_DIR=/home/ubuntu/AccelOpt

cp -r /home/ubuntu/flashinfer-trace/blob /home/ubuntu/AccelOpt/experiments/flb_interface/checkpoints

python create_folders.py --project_name 08010918 \
    --org_name zhang677 \
    --exp_date_base 12-21-17-05 \
    --traceset_root /home/ubuntu/AccelOpt/experiments/flb_interface/checkpoints

rm -rf /home/ubuntu/AccelOpt/experiments/checkpoints/12-21-17-05

python clean_checkpoints.py --chkpt_path /home/ubuntu/AccelOpt/experiments/checkpoints/12-21-17-05

**Run clean_checkpoints.py before resume_folders!!**
python resume_folders.py --project_name 08010918 \
    --org_name zhang677 \
    --exp_date_base 12-21-17-05 \
    --traceset_root /home/ubuntu/AccelOpt/experiments/flb_interface/checkpoints

PCIe or SXM (HGX): nvidia-smi -q

# Compare with full_complete_local
1. Remove the candidates.csv because profile_results.csv already stores the profiling results.

Observations:
1. gpt-oss-120b tends to just do hyperparameter tuning: How to prevent that?
2. gpt-oss-120b doesn't generate the `run` function
~~3. Where does "Unknown error" come from?~~
3. Investigate the plans
4. Why only 1 GPU is used? => Add cuda device var (only 8 kernels?) [TODO] Decide which 8 cases to use after the initial experiments 12-21-17-05
~~5. [TODO] Add a script that collects the best latency from executor_results.json~~
6. Create the baseline folder under experiments (12-21-17-05 is special. In the future experiments, the baseline traces will be in the checkpoint folder instead of flb_optimize)
7. Investigate the best kernels of 12-21-17-05

# Result Analysis for 12-21-17-05
Run `collect_best_programs.py` to externalize the Python programs from Solution jsons

gemm_n28672_k4096_32cd2698
M = 8192
N = 28672
K = 4096

Peak: 2 * M * N * K / (989 * 1e9) = 1.945 ms
Baseline: 3.294
Best: 2.85
Only autotune

fused_add_rmsnorm_h7168_d9c27791 -> No improvement
B = 14521
H = 7168
Peak: (B * H + B * H + H + B * H) * 2 / (3.35 * 1e9) = 0.186
Baseline: 0.209
Best: 0.209
(Autotune + a short-cut path skipping the for-loop)

gqa_ragged_prefill_causal_h32_kv4_d128_007ddab
Baseline: 9.418
Best: 2.695
Difference summary: https://aistudio.google.com/u/1/prompts/1lwXnvYC2L_0sw-BU15JAKTdIfkqKH2zG?pli=1 KV-head parallel

gqa_paged_decode_h32_kv4_d128_ps1_7a7fc28
Baseline: 0.444
Best: 0.261
https://aistudio.google.com/u/1/prompts/1LiEczsqFjoaHNjT3qnT31tRw9P1tyZR2 Preprocess + KV-head parallel

rmsnorm_h128_f2872f8
Baseline: 0.319
Best: 0.153
Autotune + software pipeline

mla_paged_decode_h16_ckv512_kpe64_ps1_939f995
Baseline: 1.400
Best: 0.793
https://aistudio.google.com/u/1/prompts/1ZK7283FZeUe5latzuU3CmH0GSXKxha2m Move heads from inter to intra CTA parallel + Leverage H=16

mla_paged_prefill_causal_h16_ckv512_kpe64_ps1_733a7bb
Baseline: 96.601
Best: 85.658
Loop unroll + instruction reordering

moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048_5e8dc11
Baseline: 57.468
Best: 54.218
Optimized triton kernels but tl.float8e4m3fn is never triggered





