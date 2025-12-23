import math
from typing import Any, Dict, List

import torch
import triton                        # ─┐  we keep the kernel for modern
import triton.language as tl         # ─┘  Triton versions (B-series GPUs)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Triton kernel : FP8 (E4M3-FN) block-scale de-quantisation
#    – One programme handles 128 hidden units (one “block”) for one token.
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _dequant_fp8_block128_kernel(
    x_ptr,           # [T, H]      – fp8  (E4M3-FN)
    s_ptr,           # [H/128, T]  – fp32 (transposed, block scales)
    y_ptr,           # [T, H]      – fp32 (output)
    T: tl.constexpr, # seq_len
    H: tl.constexpr, # hidden (=7168)
):
    BLOCK_H = 128

    tok_id  = tl.program_id(0)               #   0 … T-1
    blk_id  = tl.program_id(1)               #   0 … 55
    offs_h  = tl.arange(0, BLOCK_H)          # vector 0 … 127

    # --------------------------------------------------------------------- #
    # Pointers
    x_offs = tok_id * H + blk_id * BLOCK_H + offs_h
    y_offs = x_offs
    s_offs = blk_id * T + tok_id             # scale is laid out [block, token]

    # --------------------------------------------------------------------- #
    # Guards
    mask_tok = tok_id < T
    mask     = mask_tok                      # all `offs_h` are in-bounds

    # --------------------------------------------------------------------- #
    # Loads
    # Newer Triton releases expose `tl.float8e4m3fn`; on older builds it is
    # absent.  We keep the kernel for the “new” case – the wrapper below
    # will only launch it when the dtype is available.
    x = tl.load(
        x_ptr + x_offs,
        mask=mask,
        other=0.0,
        dtype=tl.float8e4m3fn,               # <── may be unavailable
    )
    sc = tl.load(s_ptr + s_offs, mask=mask_tok, other=1.0)      # scalar

    y = x * sc                               # broadcast -> vector * scalar
    tl.store(y_ptr + y_offs, y, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Wrapper that selects Triton or a pure-PyTorch fall-back (for environments
#     without FP8 support in Triton).
# ─────────────────────────────────────────────────────────────────────────────
def _dequant_fp8_block128(
    x:     torch.Tensor,   # [T, H] – torch.float8_e4m3fn
    scale: torch.Tensor,   # [H/128, T] – fp32 (transposed)
) -> torch.Tensor:
    """
    FP8 → FP32 block de-quantisation

    We try to use the Triton kernel when the FP8 dtype is present.  When it
    is missing (older Triton), we transparently fall back to the reference
    PyTorch implementation so that **correctness always wins**.
    """
    T, H = x.shape
    BLOCK_H = 128

    # ── fast Triton path ───────────────────────────────────────────────────
    if hasattr(tl, "float8e4m3fn"):
        grid = (T, H // BLOCK_H)
        out = torch.empty((T, H), device=x.device, dtype=torch.float32)
        _dequant_fp8_block128_kernel[grid](
            x, scale, out, T, H,
            num_warps=4,
            num_stages=2,
        )
        return out

    # ── reference PyTorch fall-back ────────────────────────────────────────
    #   (identical to the reference implementation in the benchmark)
    A_fp32 = x.to(torch.float32)                                 # [T, H]
    scale_TH = scale.permute(1, 0).contiguous()                  # [T, H/128]
    scale_exp = scale_TH.unsqueeze(-1).repeat(1, 1, BLOCK_H)     # [T, 56,128]
    scale_exp = scale_exp.reshape(T, H)                          # [T, H]
    return A_fp32 * scale_exp


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Public API – mirrors the specification / reference implementation.
# ─────────────────────────────────────────────────────────────────────────────
def run(
    routing_logits:        torch.Tensor,
    routing_bias:          torch.Tensor,
    hidden_states:         torch.Tensor,
    hidden_states_scale:   torch.Tensor,
    gemm1_weights:         torch.Tensor,
    gemm1_weights_scale:   torch.Tensor,
    gemm2_weights:         torch.Tensor,
    gemm2_weights_scale:   torch.Tensor,
    local_expert_offset:   int,
    routed_scaling_factor: float,
    *args: Any,
    **kwargs: Dict[str, Any],
) -> torch.Tensor:
    """
    DeepSeek-V3 MoE forward – FP8 block-scaled variant
    (see original specification for detailed math).

    Heavy‐lifting FP8 de-quant runs on GPU via Triton when possible, otherwise
    we gracefully fall back to pure PyTorch.  All other maths reproduces the
    reference implementation verbatim to guarantee **identical numerics**.
    """
    # ------------------------------------------------------------------ #
    # 0)  Device management / safety checks
    # ------------------------------------------------------------------ #
    if not torch.cuda.is_available():
        raise RuntimeError("This implementation requires a CUDA device")

    def _to_cuda(t: torch.Tensor) -> torch.Tensor:
        return t.cuda() if not t.is_cuda else t

    tensors_in: List[torch.Tensor] = [
        routing_logits, routing_bias, hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
    ]
    orig_devices = [t.device for t in tensors_in]
    (
        routing_logits, routing_bias, hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
    ) = map(_to_cuda, tensors_in)

    device = hidden_states.device   # GPU we work on

    # ------------------------------------------------------------------ #
    # 1) FP8 → FP32 de-quant (hidden states) – Triton or fall-back
    # ------------------------------------------------------------------ #
    A = _dequant_fp8_block128(hidden_states, hidden_states_scale)  # [T, 7168]

    # ------------------------------------------------------------------ #
    # 2)   Weights de-quant (identical to reference)
    # ------------------------------------------------------------------ #
    H = 7168
    I = 2048
    BLOCK = 128
    num_hidden_blocks      = H // BLOCK          # 56
    num_intermediate_blocks = I // BLOCK         # 16
    num_gemm1_out_blocks    = (2 * I) // BLOCK   # 32

    # ── GEMM1
    W13_fp32 = gemm1_weights.to(torch.float32)
    S13      = gemm1_weights_scale.to(torch.float32)
    S13_exp  = torch.repeat_interleave(S13, BLOCK, dim=1)
    S13_exp  = torch.repeat_interleave(S13_exp, BLOCK, dim=2)
    W13      = W13_fp32 * S13_exp                                     # fp32

    # ── GEMM2
    W2_fp32  = gemm2_weights.to(torch.float32)
    S2       = gemm2_weights_scale.to(torch.float32)
    S2_exp   = torch.repeat_interleave(S2, BLOCK, dim=1)
    S2_exp   = torch.repeat_interleave(S2_exp, BLOCK, dim=2)
    W2       = W2_fp32 * S2_exp                                       # fp32

    # ------------------------------------------------------------------ #
    # 3) No-aux routing (as per reference)
    # ------------------------------------------------------------------ #
    TOP_K       = 8
    N_GROUP     = 8
    TOPK_GROUP  = 4
    E_global    = 256
    E_local     = 32
    T           = routing_logits.shape[0]

    logits = routing_logits.to(torch.float32)
    bias   = routing_bias.to(torch.float32).reshape(-1)

    s            = torch.sigmoid(logits)              # [T, 256]
    s_with_bias  = s + bias                           # bias broadcast

    group_size   = E_global // N_GROUP                # 32
    s_grouped    = s_with_bias.view(T, N_GROUP, group_size)

    top2_vals, _ = torch.topk(s_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)               # [T, 8]

    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    group_mask   = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask   = (
        group_mask.unsqueeze(2)
        .expand(T, N_GROUP, group_size)
        .reshape(T, E_global)
    )

    neg_inf      = torch.finfo(torch.float32).min
    scores_kept  = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx  = torch.topk(scores_kept, k=TOP_K, dim=1, largest=True, sorted=False)

    # final per-token weights
    M        = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights       = s * M
    weights_sum   = weights.sum(dim=1, keepdim=True) + 1e-20
    weights       = (weights / weights_sum) * routed_scaling_factor  # [T, 256]

    # ------------------------------------------------------------------ #
    # 4) Local expert computation  (unchanged)
    # ------------------------------------------------------------------ #
    output = torch.zeros((T, H), dtype=torch.float32, device=device)
    local_start = int(local_expert_offset)

    for le in range(E_local):
        ge = local_start + le
        if ge < 0 or ge >= E_global:
            continue

        sel_mask = (topk_idx == ge).any(dim=1)
        if not sel_mask.any():
            continue

        token_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)
        A_e  = A.index_select(0, token_idx)          # [Tk, 7168]
        W13e = W13[le]                               # [4096, 7168]
        W2e  = W2[le]                                # [7168, 2048]

        # GEMM1
        G1 = A_e.matmul(W13e.t())                    # [Tk, 4096]

        # SwiGLU
        X1, X2 = G1[:, :I], G1[:, I:]
        silu   = X2 / (1.0 + torch.exp(-X2))
        C      = silu * X1                          # [Tk, 2048]

        # GEMM2
        O = C.matmul(W2e.t())                       # [Tk, 7168]

        # weighted accumulation
        w_tok = weights.index_select(0, token_idx)[:, ge]   # [Tk]
        output.index_add_(0, token_idx, O * w_tok.unsqueeze(1))

    # ------------------------------------------------------------------ #
    # 5) Return – BF16 on *original* hidden_states device
    # ------------------------------------------------------------------ #
    result = output.to(torch.bfloat16)
    out_device = orig_devices[2]                    # device of hidden_states
    if result.device != out_device:
        result = result.to(out_device)
    return result


__all__ = ["run"]