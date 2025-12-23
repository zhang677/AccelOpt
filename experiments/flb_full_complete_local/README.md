export ACCELOPT_BASE_DIR=/home/ubuntu/AccelOpt

python create_folders.py --project_name 08010918 \
    --org_name zhang677 \
    --exp_date_base 12-21-17-05 \
    --traceset_root /home/ubuntu/AccelOpt/experiments/flb_interface/checkpoints

rm -rf /home/ubuntu/AccelOpt/experiments/checkpoints/12-21-17-05

python clean_checkpoints.py --chkpt_path /home/ubuntu/AccelOpt/experiments/checkpoints/12-21-17-05

# Run clean_checkpoints.py before resume_folders!!
python resume_folders.py --project_name 08010918 \
    --org_name zhang677 \
    --exp_date_base 12-21-17-05 \
    --traceset_root /home/ubuntu/AccelOpt/experiments/flb_interface/checkpoints

PCIe or SXM (HGX): nvidia-smi -q

Compare with full_complete_local:
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

gemm_n28672_k4096_32cd2698
M = 8192
N = 28672
K = 4096

Peak: 2 * M * N * K / (989 * 1e9) = 1.945 ms
Baseline: 3.294
Best: 2.9 # /home/ubuntu/AccelOpt/experiments/flb_interface/checkpoints/solutions/gemm/gemm_n28672_k4096/e5926f58916a43daa59f6f5cd3d84fa6.json (Only autotune)

fused_add_rmsnorm_h7168_d9c27791 -> No improvement
B = 14521
H = 7168
Peak: (B * H + B * H + H + B * H) * 2 / (3.35 * 1e9) = 0.186
Baseline: 0.209
Best: 0.209
(Autotune + a short-cut path skipping the for-loop)

