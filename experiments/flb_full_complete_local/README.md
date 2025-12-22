export ACCELOPT_BASE_DIR=/home/ubuntu/AccelOpt

python create_folders.py --project_name 08010918 \
    --org_name zhang677 \
    --exp_date_base 12-21-17-05 \
    --traceset_root /home/ubuntu/AccelOpt/experiments/flb_interface/checkpoints

rm -rf /home/ubuntu/AccelOpt/experiments/checkpoints/12-21-17-05

python clean_checkpoints.py --chkpt_path /home/ubuntu/AccelOpt/experiments/checkpoints/12-21-17-05

PCIe or SXM (HGX): nvidia-smi -q

Compare with full_complete_local:
1. Remove the candidates.csv because profile_results.csv already stores the profiling results.

Observations:
1. gpt-oss-120b tends to just do hyperparameter tuning: How to prevent that?
2. gpt-oss-120b doesn't generate the `run` function
~~3. Where does "Unknown error" come from?~~
3. Investigate the plans
4. Why only 1 GPU is used?

gemm_n28672_k4096_32cd2698
M = 8192
N = 28672
K = 4096

Peak: 2 * M * N * K / (756 * 1e9) = 2.545 ms
Baseline: 3.294
Best: 2.9
