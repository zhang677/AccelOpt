import math
import torch
import triton
import triton.language as tl

# -----------------------------------------------------------------------------#
# Constants
# -----------------------------------------------------------------------------#
EPS: float = 1e-6          # numerical stability
HIDDEN_SIZE: int = 128     # problem-specific constant

# -----------------------------------------------------------------------------#
# Triton Kernel
# -----------------------------------------------------------------------------#
@triton.jit
def _rmsnorm_kernel(
    x_ptr,                      # [batch, hidden]  (BF16)
    w_ptr,                      # [hidden]         (BF16)
    o_ptr,                      # [batch, hidden]  (BF16)
    stride_x,                   # leading dimension of x
    stride_o,                   # leading dimension of o
    eps: tl.constexpr,          # epsilon
    hidden: tl.constexpr        # hidden size (128)
):
    pid = tl.program_id(axis=0)                         # one program = one row
    offs = tl.arange(0, hidden)                         # [0 .. 127]
    mask = offs < hidden                                # always true, kept for safety

    # -------------------------------------------------------------------------#
    # Load input row and weight vector
    # -------------------------------------------------------------------------#
    x_row_ptr = x_ptr + pid * stride_x + offs
    w_ptrs    = w_ptr + offs
    x_bf16    = tl.load(x_row_ptr, mask=mask, other=0.0)
    w_bf16    = tl.load(w_ptrs,    mask=mask, other=0.0)

    x_f32 = x_bf16.to(tl.float32)
    w_f32 = w_bf16.to(tl.float32)

    # -------------------------------------------------------------------------#
    # RMS computation
    # -------------------------------------------------------------------------#
    rsq   = x_f32 * x_f32
    mean  = tl.sum(rsq) / hidden
    inv_r = tl.rsqrt(mean + eps)

    # -------------------------------------------------------------------------#
    # Final output:  y = (x * inv_rms) * weight
    # -------------------------------------------------------------------------#
    y_f32 = (x_f32 * inv_r) * w_f32
    y_bf16 = y_f32.to(tl.bfloat16)

    # -------------------------------------------------------------------------#
    # Store
    # -------------------------------------------------------------------------#
    o_row_ptr = o_ptr + pid * stride_o + offs
    tl.store(o_row_ptr, y_bf16, mask=mask)


# -----------------------------------------------------------------------------#
# Python Wrapper
# -----------------------------------------------------------------------------#
def run(*args, **kwargs):
    """
    Entry point.

    Parameters (positional or keyword):
      hidden_states: Tensor[batch, 128] (bfloat16)
      weight:        Tensor[128]        (bfloat16)

    Returns:
      output Tensor with same shape/dtype/device as `hidden_states`
    """
    # -------------------------------------------------------------------------#
    # Argument extraction
    # -------------------------------------------------------------------------#
    if len(args) + len(kwargs) < 2:
        raise TypeError("run() missing required arguments: 'hidden_states' and 'weight'")

    hidden_states = kwargs.pop('hidden_states') if 'hidden_states' in kwargs else args[0]
    weight        = kwargs.pop('weight')        if 'weight'        in kwargs else args[1] if len(args) > 1 else None
    if weight is None:
        raise TypeError("run() missing required argument: 'weight'")
    if kwargs:
        raise TypeError(f"run() got unexpected keyword arguments {list(kwargs.keys())}")

    # -------------------------------------------------------------------------#
    # Shape / dtype checks
    # -------------------------------------------------------------------------#
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must be 2-D [batch, hidden]")
    batch, hidden = hidden_states.shape
    if hidden != HIDDEN_SIZE:
        raise ValueError(f"hidden dimension must be {HIDDEN_SIZE}")
    if weight.shape != (HIDDEN_SIZE,):
        raise ValueError(f"weight shape must be ({HIDDEN_SIZE},)")

    # -------------------------------------------------------------------------#
    # Device handling
    # -------------------------------------------------------------------------#
    if not torch.cuda.is_available():
        if hidden_states.is_cuda or weight.is_cuda:
            raise RuntimeError("CUDA tensors provided but CUDA is not available")
        # CPU fallback (reference implementation)
        x = hidden_states.to(torch.float32)
        inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
        y = (x * inv_rms) * weight.to(torch.float32)
        return y.to(hidden_states.dtype)

    orig_device = hidden_states.device
    x_gpu = hidden_states if hidden_states.is_cuda else hidden_states.cuda()
    w_gpu = weight        if weight.is_cuda        else weight.cuda()

    # Ensure contiguous layout for predictable strides
    x_gpu = x_gpu.contiguous()
    w_gpu = w_gpu.contiguous()

    # Allocate output
    o_gpu = torch.empty_like(x_gpu)

    # -------------------------------------------------------------------------#
    # Kernel launch
    # -------------------------------------------------------------------------#
    grid = (batch,)
    _rmsnorm_kernel[grid](
        x_gpu, w_gpu, o_gpu,
        x_gpu.stride(0), o_gpu.stride(0),
        EPS, HIDDEN_SIZE,
        num_warps=4
    )

    # -------------------------------------------------------------------------#
    # Move back to original device if necessary
    # -------------------------------------------------------------------------#
    if orig_device.type == 'cpu':
        return o_gpu.cpu()
    return o_gpu


# -----------------------------------------------------------------------------#
# This file exposes a single callable `run` for external use
# -----------------------------------------------------------------------------#
__all__ = ["run"]