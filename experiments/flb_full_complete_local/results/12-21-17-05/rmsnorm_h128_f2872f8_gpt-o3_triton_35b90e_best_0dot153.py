# =============================================================================
# rmsnorm_h128 – Triton implementation (batch‑tiling, weight‑reuse,
#                triple‑buffered row pre‑fetch, pipelined)
# =============================================================================
import torch
import triton
import triton.language as tl

# --------------------------------------------------------------------------- #
# Constants (must stay BF16 as in the reference implementation)
# --------------------------------------------------------------------------- #
EPS: float = 1e-6          # numerical‑stability term
HIDDEN_SIZE: int = 128     # hidden dimension (fixed)

# --------------------------------------------------------------------------- #
# Triton kernel – autotuned over ROWS_PER_BLOCK, num_warps and num_stages
# --------------------------------------------------------------------------- #
@triton.autotune(
    configs=[
        # ---- original configurations (kept) -------------------------------- #
        triton.Config({"BLOCK": HIDDEN_SIZE, "ROWS_PER_BLOCK": 1, "num_warps": 4}, num_stages=2),
        triton.Config({"BLOCK": HIDDEN_SIZE, "ROWS_PER_BLOCK": 1, "num_warps": 4}, num_stages=3),
        triton.Config({"BLOCK": HIDDEN_SIZE, "ROWS_PER_BLOCK": 1, "num_warps": 4}, num_stages=4),
        triton.Config({"BLOCK": HIDDEN_SIZE, "ROWS_PER_BLOCK": 1, "num_warps": 4}, num_stages=5),

        triton.Config({"BLOCK": HIDDEN_SIZE, "ROWS_PER_BLOCK": 2, "num_warps": 4}, num_stages=2),
        triton.Config({"BLOCK": HIDDEN_SIZE, "ROWS_PER_BLOCK": 2, "num_warps": 4}, num_stages=3),
        triton.Config({"BLOCK": HIDDEN_SIZE, "ROWS_PER_BLOCK": 2, "num_warps": 4}, num_stages=4),
        triton.Config({"BLOCK": HIDDEN_SIZE, "ROWS_PER_BLOCK": 2, "num_warps": 4}, num_stages=5),

        triton.Config({"BLOCK": HIDDEN_SIZE, "ROWS_PER_BLOCK": 4, "num_warps": 8}, num_stages=2),
        triton.Config({"BLOCK": HIDDEN_SIZE, "ROWS_PER_BLOCK": 4, "num_warps": 8}, num_stages=3),
        triton.Config({"BLOCK": HIDDEN_SIZE, "ROWS_PER_BLOCK": 4, "num_warps": 8}, num_stages=4),
        triton.Config({"BLOCK": HIDDEN_SIZE, "ROWS_PER_BLOCK": 4, "num_warps": 8}, num_stages=5),

        triton.Config({"BLOCK": HIDDEN_SIZE, "ROWS_PER_BLOCK": 8, "num_warps": 8}, num_stages=2),
        triton.Config({"BLOCK": HIDDEN_SIZE, "ROWS_PER_BLOCK": 8, "num_warps": 8}, num_stages=3),
        triton.Config({"BLOCK": HIDDEN_SIZE, "ROWS_PER_BLOCK": 8, "num_warps": 8}, num_stages=4),
        triton.Config({"BLOCK": HIDDEN_SIZE, "ROWS_PER_BLOCK": 8, "num_warps": 8}, num_stages=5),

        # ---- new, larger ROWS_PER_BLOCK values ------------------------------ #
        triton.Config({"BLOCK": HIDDEN_SIZE, "ROWS_PER_BLOCK": 12, "num_warps": 8},  num_stages=3),
        triton.Config({"BLOCK": HIDDEN_SIZE, "ROWS_PER_BLOCK": 16, "num_warps": 8},  num_stages=3),
        triton.Config({"BLOCK": HIDDEN_SIZE, "ROWS_PER_BLOCK": 24, "num_warps": 8},  num_stages=4),
        triton.Config({"BLOCK": HIDDEN_SIZE, "ROWS_PER_BLOCK": 32, "num_warps": 16}, num_stages=4),
        triton.Config({"BLOCK": HIDDEN_SIZE, "ROWS_PER_BLOCK": 64, "num_warps": 16}, num_stages=5),
    ],
    key=["hidden"],   # hidden size is the only runtime‑varying dimension
)
@triton.jit
def _rmsnorm_kernel(
    x_ptr,          # [batch, hidden]    (bf16)
    w_ptr,          # [hidden]           (bf16)
    o_ptr,          # [batch, hidden]    (bf16)
    stride_x,       # stride between rows in x
    stride_o,       # stride between rows in o
    batch,          # total number of rows (runtime int)
    eps: tl.constexpr,        # epsilon for numerical stability
    hidden: tl.constexpr,     # hidden size (128)
    ROWS_PER_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Triple‑buffered RMSNorm.
    Each program processes ``ROWS_PER_BLOCK`` consecutive rows.
    Three register buffers keep rows i, i+1 and i+2 in flight,
    so that the load of row i+3 can be started while row i is still being reduced.
    """
    pid = tl.program_id(axis=0)               # tile index
    base_row = pid * ROWS_PER_BLOCK            # first row handled by this program

    # ------------------------------------------------------------------- #
    # Offsets & mask for hidden dimension (BLOCK == hidden)
    # ------------------------------------------------------------------- #
    offs = tl.arange(0, BLOCK)
    hidden_mask = offs < hidden

    # ------------------------------------------------------------------- #
    # Load weight once (bf16 → fp32)
    # ------------------------------------------------------------------- #
    w_bf16 = tl.load(w_ptr + offs, mask=hidden_mask, other=0.0)
    w_f32 = w_bf16.to(tl.float32)

    # ------------------------------------------------------------------- #
    # Triple‑buffer pre‑load: rows base, base+1, base+2
    # ------------------------------------------------------------------- #
    r0 = base_row + 0
    r1 = base_row + 1
    r2 = base_row + 2

    m0 = hidden_mask & (r0 < batch)
    m1 = hidden_mask & (r1 < batch)
    m2 = hidden_mask & (r2 < batch)

    buf0 = tl.load(x_ptr + r0 * stride_x + offs, mask=m0, other=0.0).to(tl.float32)
    buf1 = tl.load(x_ptr + r1 * stride_x + offs, mask=m1, other=0.0).to(tl.float32)
    buf2 = tl.load(x_ptr + r2 * stride_x + offs, mask=m2, other=0.0).to(tl.float32)

    # ------------------------------------------------------------------- #
    # Main loop – each iteration consumes the oldest buffer and pre‑fetches
    # the row that will be needed three steps later (i+3)
    # ------------------------------------------------------------------- #
    for i in range(ROWS_PER_BLOCK):
        cur_row = base_row + i
        cur_mask = hidden_mask & (cur_row < batch)

        # ---- select the correct buffer -------------------------------------------------
        if i % 3 == 0:
            cur_buf = buf0
        elif i % 3 == 1:
            cur_buf = buf1
        else:               # i % 3 == 2
            cur_buf = buf2

        # ---- RMS‑Norm computation ------------------------------------------------------
        rsq   = cur_buf * cur_buf
        mean  = tl.sum(rsq) / hidden
        inv_r = tl.rsqrt(mean + eps)
        y_f32 = (cur_buf * inv_r) * w_f32
        y_bf16 = y_f32.to(tl.bfloat16)

        # ---- store --------------------------------------------------------------------
        o_ptr_cur = o_ptr + cur_row * stride_o + offs
        tl.store(o_ptr_cur, y_bf16, mask=cur_mask)

        # ---- pre‑fetch row (cur_row + 3) into the buffer we have just used -------------
        next_row = cur_row + 3
        next_mask = hidden_mask & (next_row < batch)

        if i % 3 == 0:
            # we just used buf0 → refill it
            buf0 = tl.load(x_ptr + next_row * stride_x + offs,
                           mask=next_mask, other=0.0).to(tl.float32)
        elif i % 3 == 1:
            buf1 = tl.load(x_ptr + next_row * stride_x + offs,
                           mask=next_mask, other=0.0).to(tl.float32)
        else:
            buf2 = tl.load(x_ptr + next_row * stride_x + offs,
                           mask=next_mask, other=0.0).to(tl.float32)


# --------------------------------------------------------------------------- #
# Python wrapper – identical API to the reference implementation
# --------------------------------------------------------------------------- #
def run(*args, **kwargs):
    """
    Triton‑accelerated RMSNorm for hidden_size = 128 (bf16) with
    batch‑tiling, weight‑reuse and a triple‑buffered software pipeline.

    Parameters (positional or keyword):
        hidden_states : torch.Tensor[batch, 128] (bfloat16)
        weight        : torch.Tensor[128]      (bfloat16)

    Returns:
        torch.Tensor[batch, 128] (bfloat16) – on the same device as ``hidden_states``.
    """
    # --------------------------------------------------------------- #
    # Argument handling
    # --------------------------------------------------------------- #
    if len(args) + len(kwargs) < 2:
        raise TypeError(
            "run() missing required arguments: 'hidden_states' and 'weight'"
        )
    hidden_states = kwargs.pop("hidden_states") if "hidden_states" in kwargs else args[0]
    weight = kwargs.pop("weight") if "weight" in kwargs else (
        args[1] if len(args) > 1 else None
    )
    if weight is None:
        raise TypeError("run() missing required argument: 'weight'")
    if kwargs:
        raise TypeError(f"run() got unexpected keyword arguments {list(kwargs.keys())}")

    # --------------------------------------------------------------- #
    # Shape / dtype checks
    # --------------------------------------------------------------- #
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must be a 2‑D tensor [batch, hidden]")
    batch, hidden = hidden_states.shape
    if hidden != HIDDEN_SIZE:
        raise ValueError(f"hidden dimension must be {HIDDEN_SIZE}")
    if weight.shape != (HIDDEN_SIZE,):
        raise ValueError(f"weight shape must be ({HIDDEN_SIZE},)")
    if hidden_states.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("both inputs must be of dtype torch.bfloat16")

    # --------------------------------------------------------------- #
    # Fallback to PyTorch when CUDA is unavailable
    # --------------------------------------------------------------- #
    if not torch.cuda.is_available():
        x = hidden_states.to(torch.float32)
        inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
        y = (x * inv_rms) * weight.to(torch.float32)
        return y.to(hidden_states.dtype)

    # --------------------------------------------------------------- #
    # Prepare contiguous GPU tensors and allocate output
    # --------------------------------------------------------------- #
    device = hidden_states.device
    x_gpu = hidden_states if hidden_states.is_cuda else hidden_states.to(device)
    w_gpu = weight if weight.is_cuda else weight.to(device)

    x_gpu = x_gpu.contiguous()
    w_gpu = w_gpu.contiguous()
    out_gpu = torch.empty_like(x_gpu)

    # --------------------------------------------------------------- #
    # Kernel launch – grid = ceil(batch / ROWS_PER_BLOCK)
    # --------------------------------------------------------------- #
    grid = lambda META: ((batch + META["ROWS_PER_BLOCK"] - 1) // META["ROWS_PER_BLOCK"],)
    _rmsnorm_kernel[grid](
        x_gpu,
        w_gpu,
        out_gpu,
        x_gpu.stride(0),
        out_gpu.stride(0),
        batch,
        EPS,
        HIDDEN_SIZE,
        # ROWS_PER_BLOCK and BLOCK are injected automatically from the chosen config
    )

    # --------------------------------------------------------------- #
    # Return tensor on the original device
    # --------------------------------------------------------------- #
    return out_gpu if device.type == "cuda" else out_gpu.cpu()


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
__all__ = ["run"]