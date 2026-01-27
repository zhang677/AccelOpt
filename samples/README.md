# Mamba
`mamba_v1.py`, `mamba_v2.py`, and `mamba_v3.py` are adopted from [NKI tutorials](https://github.com/aws-neuron/nki-samples/blob/main/src/nki_samples/tutorials/fused_mamba/mamba_nki_kernels.py).

`mamba_optimized.py` is the best Mamba kernel AccelOpt has discovered.

`mamba_v1.py` is the same as `../NKIBench/kernels/mamba_M7168_C256_S16_0.py`

`profile_mambas.py` profiles all these 4 kernels.

# RoPE
The baseline kernel is in `../NKIBench/kernels/rope_single_freq_apply_B1_H64_N4096_D128_0.py` adopted from [NKI samples](https://github.com/aws-neuron/nki-samples/blob/main/src/nki_samples/tutorials/rotary/rotary_nki_kernels.py).

`rope_optimized.py` is the best RoPE kernel AccelOpt has discovered.

`profile_ropes.py` profiles both kernels.

The actual performance might be sightly different from numbers we reported in the rebuttal response because of the fluctuation.