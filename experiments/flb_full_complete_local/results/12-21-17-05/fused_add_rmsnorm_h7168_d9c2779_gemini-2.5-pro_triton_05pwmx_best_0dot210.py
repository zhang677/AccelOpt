import torch
import triton
import triton.language as tl

# --------------------------------------------------------------------
# Kernel: one‑pass fused add + RMSNorm + weight scaling
# --------------------------------------------------------------------
@triton.autotune(
    configs=[
        # The only sensible config for the fixed hidden size 7168 is to
        # cover the whole row in one iteration (BLOCK_SIZE_N ≥ HIDDEN_SIZE).
        triton.Config({'BLOCK_SIZE_N': 8192, 'num_warps': 16, 'num_stages': 2}),
        # Keeping the other configs would lead to incorrect results because
        # they would not process the whole row.  The autotuner will therefore
        # always pick the 8192‑element configuration.
    ],
    key=['HIDDEN_SIZE'],
)
@triton.jit
def _fused_add_rmsnorm_h7168_kernel(
    hidden_states_ptr,        # const bfloat16 *
    residual_ptr,             # const bfloat16 *
    weight_ptr,               # const bfloat16 *
    output_ptr,               # bfloat16 *
    stride_hs_b,              # int64
    stride_res_b,             # int64
    stride_out_b,             # int64
    HIDDEN_SIZE: tl.constexpr,   # = 7168 (compile‑time)
    EPS: tl.constexpr,            # = 1e‑6  (compile‑time)
    BLOCK_SIZE_N: tl.constexpr,   # must be ≥ HIDDEN_SIZE
):
    """
    One‑pass kernel:
      1) Load the whole row of `hidden_states` and `residual` (masked).
      2) Compute x = hidden_states + residual.
      3) Reduce x² to obtain the variance.
      4) Compute inv_rms = rsqrt(variance + EPS).
      5) Load the corresponding slice of `weight`.
      6) Write out (x * inv_rms * weight) as bfloat16.
    All arithmetic is performed in float32, preserving the baseline precision.
    """
    pid = tl.program_id(axis=0)                     # one program per batch element

    # Pointers to the current row
    row_hs = hidden_states_ptr + pid * stride_hs_b
    row_res = residual_ptr      + pid * stride_res_b
    row_out = output_ptr        + pid * stride_out_b

    # ----------------------------------------------------------------
    # 1) Load the whole row (masked – the extra elements are zeroed)
    # ----------------------------------------------------------------
    offs = tl.arange(0, BLOCK_SIZE_N)
    mask = offs < HIDDEN_SIZE

    hs = tl.load(row_hs + offs, mask=mask, other=0.0).to(tl.float32)
    rs = tl.load(row_res + offs, mask=mask, other=0.0).to(tl.float32)

    # ----------------------------------------------------------------
    # 2) Fused add
    # ----------------------------------------------------------------
    x = hs + rs                     # shape: [BLOCK_SIZE_N] (float32)

    # ----------------------------------------------------------------
    # 3) RMS computation (block‑wide reduction)
    # ----------------------------------------------------------------
    variance = tl.sum(x * x, axis=0) / HIDDEN_SIZE
    inv_rms  = tl.rsqrt(variance + EPS)

    # ----------------------------------------------------------------
    # 4) Apply weight and store
    # ----------------------------------------------------------------
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = x * inv_rms * w               # still float32

    tl.store(row_out + offs, out.to(tl.bfloat16), mask=mask)


# --------------------------------------------------------------------
# Python wrapper – validates inputs, launches the kernel and returns result
# --------------------------------------------------------------------
def run(hidden_states: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor) -> torch.Tensor:
    """
    Fused add + RMSNorm (hidden size = 7168) using the optimized Triton kernel.
    The function keeps the exact numerical behavior of the reference
    implementation (float32 intermediate arithmetic, bfloat16 I/O).

    Args:
        hidden_states: Tensor[batch, 7168], dtype=torch.bfloat16
        residual:      Tensor[batch, 7168], dtype=torch.bfloat16
        weight:        Tensor[7168],       dtype=torch.bfloat16

    Returns:
        Tensor[batch, 7168], dtype=torch.bfloat16
    """
    # --------------------------------------------------------------
    # 1️⃣ Input validation & device handling
    # --------------------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError("Triton kernels require a CUDA‑enabled GPU.")

    # Make sure all tensors are on the same CUDA device (move if needed)
    device = hidden_states.device
    if device.type != "cuda":
        device = torch.device("cuda")
    hidden_states = hidden_states.to(device)
    residual = residual.to(device)
    weight = weight.to(device)

    if hidden_states.dtype != torch.bfloat16 or \
       residual.dtype      != torch.bfloat16 or \
       weight.dtype        != torch.bfloat16:
        raise TypeError("All inputs must be torch.bfloat16 tensors.")

    batch_size, hidden_size = hidden_states.shape
    if hidden_size != 7168:
        raise ValueError(f"hidden_size must be 7168, got {hidden_size}")
    if residual.shape != hidden_states.shape:
        raise ValueError("`residual` must have the same shape as `hidden_states`.")
    if weight.shape != (hidden_size,):
        raise ValueError(f"`weight` must be of shape ({hidden_size},).")

    # --------------------------------------------------------------
    # 2️⃣ Output allocation
    # --------------------------------------------------------------
    output = torch.empty_like(hidden_states)

    # --------------------------------------------------------------
    # 3️⃣ Grid configuration (one program per batch element)
    # --------------------------------------------------------------
    grid = (batch_size,)

    # --------------------------------------------------------------
    # 4️⃣ Kernel launch
    # --------------------------------------------------------------
    _fused_add_rmsnorm_h7168_kernel[grid](
        hidden_states,
        residual,
        weight,
        output,
        hidden_states.stride(0),   # stride between rows (batch dimension)
        residual.stride(0),
        output.stride(0),
        HIDDEN_SIZE=hidden_size,
        EPS=1e-6,
        # BLOCK_SIZE_N is a compile‑time constant taken from the autotune config
    )

    return output