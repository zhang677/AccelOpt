# -------------------------------------------------------------
# 1. Imports
# -------------------------------------------------------------
import math
from typing import Any, Dict, List

import torch
import triton
import triton.language as tl

# -------------------------------------------------------------
# 2. Helper – block‑wise de‑quantisation (fallback when float‑8 not
#    available in the current Triton build)
# -------------------------------------------------------------
def _fallback_dequant_fp8_block128(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Fallback de‑quantisation (FP8 → FP32) using block‑wise scales."""
    BLOCK = 128
    x_fp32 = x.to(torch.float32)                                 # [T, H]
    s_T = scale.permute(1, 0).contiguous()                       # [T, H/128]
    s_exp = s_T.unsqueeze(-1).repeat(1, 1, BLOCK)                # [T, H/128, 128]
    s_exp = s_exp.reshape(x_fp32.shape)                         # [T, H]
    return x_fp32 * s_exp


# -------------------------------------------------------------
# 3. Triton fused kernel – token‑tiling version
# -------------------------------------------------------------
@triton.jit
def _moe_fp8_fused_token_kernel(
    # -----------------------------------------------------------------
    # pointers
    # -----------------------------------------------------------------
    hidden_ptr,          # [T, H]               fp8_e4m3fn
    hidden_sc_ptr,       # [H/128, T]           fp32   (block scales, transposed)
    w13_ptr,             # [E_local, 2*I, H]    fp8
    w13_sc_ptr,          # [E_local, (2I)/128, H/128] fp32
    w2_ptr,              # [E_local, H, I]      fp8
    w2_sc_ptr,           # [E_local, H/128, I/128]      fp32
    topk_expert_ptr,     # [T, TOP_K]           int32
    topk_weight_ptr,     # [T, TOP_K]           fp32
    out_ptr,             # [T, H]               fp32 (accumulator)
    # -----------------------------------------------------------------
    # compile‑time constants
    # -----------------------------------------------------------------
    T:                 tl.constexpr,
    H:                 tl.constexpr,
    I:                 tl.constexpr,
    E_LOCAL:           tl.constexpr,
    TOP_K:             tl.constexpr,
    BLOCK_H:           tl.constexpr = 128,
    BLOCK_I:           tl.constexpr = 128,
    TOKENS_PER_BLOCK: tl.constexpr = 2,            # <-- tiled token dimension
):
    """
    One program processes **TOKENS_PER_BLOCK** consecutive tokens.
    All expert‑weight tiles are loaded once per token‑tile and reused for
    every token inside the tile.
    """

    pid = tl.program_id(0)                     # tile id
    token_base = pid * TOKENS_PER_BLOCK
    # token indices for this tile
    token_off = token_base + tl.arange(0, TOKENS_PER_BLOCK)
    mask_token = token_off < T                  # [TOKENS_PER_BLOCK]

    # -----------------------------------------------------------------
    # 0) per‑token accumulators for the final output (registered)
    # -----------------------------------------------------------------
    out_acc = tl.zeros([TOKENS_PER_BLOCK, H], dtype=tl.float32)

    # -----------------------------------------------------------------
    # 1) loop over the TOP_K experts selected for each token
    # -----------------------------------------------------------------
    for slot in range(TOP_K):
        # ----- load the expert id and the routing weight for every token
        #      in the tile (scalar per token)
        linear_idx = token_off * TOP_K + slot               # [TOKENS_PER_BLOCK]
        expert = tl.load(topk_expert_ptr + linear_idx,
                        mask=mask_token, other=-1)         # int32
        weight = tl.load(topk_weight_ptr + linear_idx,
                        mask=mask_token, other=0.0)         # fp32

        # skip the whole slot if no token in the tile has a valid expert
        valid_slot = (expert >= 0) & (weight != 0.0) & mask_token
        if tl.any(valid_slot) == 0:
            continue

        # -----------------------------------------------------------------
        # 2) GEMM‑1 : hidden (TOKENS_PER_BLOCK × H) × W13ᵀ (2I × H)
        # -----------------------------------------------------------------
        # accumulator for the 2*I intermediate
        g1 = tl.zeros([TOKENS_PER_BLOCK, 2 * I], dtype=tl.float32)

        # number of hidden blocks
        num_h_blocks = H // BLOCK_H

        for hb in range(num_h_blocks):
            # ----- hidden slice for every token in the tile -----
            h_off   = hb * BLOCK_H + tl.arange(0, BLOCK_H)
            h_ptr   = hidden_ptr + token_off[:, None] * H + h_off[None, :]
            h_fp8   = tl.load(h_ptr, mask=mask_token[:, None], other=tl.float8e4m3fn(0))
            # block‑scale (layout: [H/128, T] – transposed)
            sc_off  = hb * T + token_off
            sc      = tl.load(hidden_sc_ptr + sc_off,
                              mask=mask_token, other=1.0)          # fp32
            h_fp32  = h_fp8.to(tl.float32) * sc[:, None]           # [TOKENS_PER_BLOCK, BLOCK_H]

            # ----- W13 tile (expert‑specific) -----
            # We load the tile **once per expert** (the same for every token in the tile)
            # but we have to respect that different tokens may point to different experts.
            # Therefore we materialise a small per‑token weight tile.
            w13_tile = tl.zeros([TOKENS_PER_BLOCK, 2 * I, BLOCK_H], dtype=tl.float32)

            for t in range(TOKENS_PER_BLOCK):
                if not mask_token[t] or expert[t] < 0:
                    continue
                # base offset for this expert
                w13_base = expert[t] * (2 * I) * H
                i_off   = tl.arange(0, 2 * I)
                w13_off = w13_base + i_off[:, None] * H + (hb * BLOCK_H + tl.arange(0, BLOCK_H))[None, :]
                w13_fp8 = tl.load(w13_ptr + w13_off,
                                  mask=True, other=tl.float8e4m3fn(0))
                # per‑block scales for W13
                blk_i   = i_off // BLOCK_H
                sc_off  = expert[t] * ((2 * I) // BLOCK_H) * (H // BLOCK_H) \
                          + blk_i * (H // BLOCK_H) + hb
                w13_sc  = tl.load(w13_sc_ptr + sc_off, mask=True, other=1.0)   # [2I/128]
                w13_sc_tile = tl.broadcast_to(w13_sc[:, None], [2 * I, BLOCK_H])
                w13_fp32 = w13_fp8.to(tl.float32) * w13_sc_tile
                w13_tile[t, :, :] = w13_fp32

            # ----- batched dot product (hidden slice × W13ᵀ) -----
            # g1[t] += hidden[t] @ W13ᵀ
            for t in range(TOKENS_PER_BLOCK):
                if not mask_token[t] or expert[t] < 0:
                    continue
                part = tl.dot(h_fp32[t, None, :], w13_tile[t, :, :])   # (1, 2I)
                g1[t, :] += part[0]

        # -----------------------------------------------------------------
        # 3) SwiGLU (still per‑token)
        # -----------------------------------------------------------------
        x1 = g1[:, 0:I]                     # [TOKENS_PER_BLOCK, I]
        x2 = g1[:, I:2 * I]                 # [TOKENS_PER_BLOCK, I]
        silu = x2 / (1.0 + tl.exp(-x2))
        glu = silu * x1                     # [TOKENS_PER_BLOCK, I]

        # -----------------------------------------------------------------
        # 4) GEMM‑2 : glu (TOKENS_PER_BLOCK × I) × W2ᵀ (H × I)
        # -----------------------------------------------------------------
        o = tl.zeros([TOKENS_PER_BLOCK, H], dtype=tl.float32)

        for hb in range(num_h_blocks):
            # ----- W2 tile (expert‑specific) -----
            w2_tile = tl.zeros([TOKENS_PER_BLOCK, BLOCK_H, I], dtype=tl.float32)

            for t in range(TOKENS_PER_BLOCK):
                if not mask_token[t] or expert[t] < 0:
                    continue
                w2_base = expert[t] * H * I
                h_off   = hb * BLOCK_H + tl.arange(0, BLOCK_H)
                i_off   = tl.arange(0, I)
                w2_off  = w2_base + h_off[:, None] + i_off[None, :] * H
                w2_fp8  = tl.load(w2_ptr + w2_off,
                                  mask=True, other=tl.float8e4m3fn(0))
                # per‑block scales for W2
                blk_i   = i_off // BLOCK_I
                sc_off  = expert[t] * (H // BLOCK_H) * (I // BLOCK_I) \
                          + hb * (I // BLOCK_I) + blk_i
                w2_sc   = tl.load(w2_sc_ptr + sc_off, mask=True, other=1.0)   # [I/128]
                w2_sc_tile = tl.broadcast_to(w2_sc[:, None], [I, BLOCK_H])
                w2_fp32 = w2_fp8.to(tl.float32) * w2_sc_tile      # (I, BLOCK_H)
                w2_tile[t, :, :] = w2_fp32

            # ----- batched dot (glu × W2ᵀ) -----
            for t in range(TOKENS_PER_BLOCK):
                if not mask_token[t] or expert[t] < 0:
                    continue
                part2 = tl.dot(glu[t, None, :], w2_tile[t, :, :])   # (1, BLOCK_H)
                o[t, hb * BLOCK_H: hb * BLOCK_H + BLOCK_H] = part2[0]

        # -----------------------------------------------------------------
        # 5) Weighted accumulation into the per‑token output accumulator
        # -----------------------------------------------------------------
        for t in range(TOKENS_PER_BLOCK):
            if not mask_token[t] or expert[t] < 0:
                continue
            out_acc[t, :] += o[t, :] * weight[t]

    # -----------------------------------------------------------------
    # 6) Write the accumulated result back to global memory
    # -----------------------------------------------------------------
    for t in range(TOKENS_PER_BLOCK):
        if not mask_token[t]:
            continue
        out_off = (token_base + t) * H + tl.arange(0, H)
        tl.store(out_ptr + out_off, out_acc[t, :], mask=True)


# -------------------------------------------------------------
# 4. Triton autotune configuration (includes token‑tiling)
# -------------------------------------------------------------
_moe_fp8_fused_token = triton.autotune(
    configs=[
        # 2 tokens per block – low shared‑mem pressure, high occupancy
        triton.Config({'BLOCK_H': 128, 'BLOCK_I': 128, 'TOKENS_PER_BLOCK': 2},
                      num_warps=4, num_stages=2, num_ctas=2),
        # 4 tokens per block – may fit on newer GPUs (e.g. H100)
        triton.Config({'BLOCK_H': 128, 'BLOCK_I': 128, 'TOKENS_PER_BLOCK': 4},
                      num_warps=8, num_stages=3, num_ctas=2),
    ],
    key=['T', 'H']
)(_moe_fp8_fused_token_kernel)


# -------------------------------------------------------------
# 5. Public API – optimised run()
# -------------------------------------------------------------
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
    Optimised DeepSeek‑V3 MoE forward pass.
    * FP8 block‑scale de‑quantisation, GEMM1 → SwiGLU → GEMM2 are fused in a
      tiled‑token Triton kernel.
    * Routing (group‑wise top‑k) matches the reference implementation
      bit‑for‑bit.
    * Returns the MoE output in ``bfloat16`` on the original device of
      ``hidden_states``.
    * Falls back to the pure‑PyTorch implementation when the current Triton
      build lacks ``float8`` support.
    """
    # -----------------------------------------------------------------
    # 0) sanity & device handling
    # -----------------------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError("MoE kernel requires a CUDA device")

    def _to_cuda(t: torch.Tensor) -> torch.Tensor:
        return t.cuda() if not t.is_cuda else t

    tensors_in: List[torch.Tensor] = [
        routing_logits, routing_bias, hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale,
        gemm2_weights, gemm2_weights_scale,
    ]
    orig_devices = [t.device for t in tensors_in]
    (
        routing_logits, routing_bias, hidden_states, hidden_states_scale,
        gemm1_weights, gemm1_weights_scale,
        gemm2_weights, gemm2_weights_scale,
    ) = map(_to_cuda, tensors_in)

    device = hidden_states.device
    T, H = hidden_states.shape          # T = seq_len, H = hidden_size (7168)
    I = 2048
    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4
    E_GLOBAL = 256
    E_LOCAL = gemm1_weights.shape[0]   # should be 32

    # -----------------------------------------------------------------
    # 1) Routing – identical to reference
    # -----------------------------------------------------------------
    logits = routing_logits.to(torch.float32)               # [T, 256]
    bias   = routing_bias.to(torch.float32).reshape(-1)     # [256]

    s = torch.sigmoid(logits)                              # [T, 256]
    s_wbias = s + bias                                     # broadcast

    group_size = E_GLOBAL // N_GROUP                        # 32
    s_grouped = s_wbias.view(T, N_GROUP, group_size)       # [T,8,32]

    top2_vals, _ = torch.topk(s_grouped, k=2, dim=2, largest=True, sorted=False)
    group_scores = top2_vals.sum(dim=2)                     # [T,8]

    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1,
                              largest=True, sorted=False)   # [T,4]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)

    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size)\
                                    .reshape(T, E_GLOBAL)   # [T,256]

    neg_inf = torch.finfo(torch.float32).min
    scores_kept = s_wbias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_kept, k=TOP_K, dim=1,
                             largest=True, sorted=False)      # [T,8] int64

    # routing weights (use s without bias)
    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights = (weights / (weights.sum(dim=1, keepdim=True) + 1e-20)) \
                * routed_scaling_factor                       # [T,256]

    # -----------------------------------------------------------------
    # 2) Kernel launch – FP8 path vs fallback
    # -----------------------------------------------------------------
    if hasattr(tl, "float8e4m3fn"):
        # ----- pack routing tensors for the kernel -----
        topk_expert = topk_idx.to(torch.int32).contiguous()          # [T, TOP_K]
        # per‑token‑expert weight = weight for the selected expert
        topk_weight = weights.gather(1, topk_idx).contiguous()      # [T, TOP_K]

        # ----- output accumulator (fp32) -----
        out = torch.zeros((T, H), dtype=torch.float32, device=device)

        # ----- launch fused kernel (grid = one tile per program) -----
        tiles = math.ceil(T / 2)          # default token‑tile = 2 (autotune may pick 4)
        grid = (tiles,)

        _moe_fp8_fused_token[grid](
            hidden_states,                     # fp8
            hidden_states_scale,               # fp32 block scales
            gemm1_weights,                     # fp8
            gemm1_weights_scale,               # fp32 scales
            gemm2_weights,                     # fp8
            gemm2_weights_scale,               # fp32 scales
            topk_expert,                       # int32
            topk_weight,                       # fp32
            out,                               # fp32 accumulator
            T=T,
            H=H,
            I=I,
            E_LOCAL=E_LOCAL,
            TOP_K=TOP_K,
            # compile‑time constants are left to autotune
            num_warps=8,          # overridden by autotune
            num_stages=3,
        )
    else:
        # ----- pure‑PyTorch fallback (identical to reference) -----
        A = _fallback_dequant_fp8_block128(hidden_states, hidden_states_scale)

        S13 = gemm1_weights_scale.to(torch.float32)
        S13 = torch.repeat_interleave(S13, 128, dim=1)
        S13 = torch.repeat_interleave(S13, 128, dim=2)
        W13 = gemm1_weights.to(torch.float32) * S13

        S2 = gemm2_weights_scale.to(torch.float32)
        S2 = torch.repeat_interleave(S2, 128, dim=1)
        S2 = torch.repeat_interleave(S2, 128, dim=2)
        W2 = gemm2_weights.to(torch.float32) * S2

        out = torch.zeros((T, H), dtype=torch.float32, device=device)
        local_start = int(local_expert_offset)

        for le in range(E_LOCAL):
            ge = local_start + le
            if ge < 0 or ge >= E_GLOBAL:
                continue
            sel_mask = (topk_idx == ge).any(dim=1)
            if not sel_mask.any():
                continue
            token_idx = torch.nonzero(sel_mask, as_tuple=False).squeeze(1)
            A_e = A.index_select(0, token_idx)          # [Tk, H]
            W13e = W13[le]                              # [4096, H]
            W2e = W2[le]                                # [H, 2048]

            G1 = A_e @ W13e.t()                         # [Tk, 4096]
            X1, X2 = G1[:, :I], G1[:, I:]
            silu = X2 / (1.0 + torch.exp(-X2))
            C = silu * X1                               # [Tk, 2048]
            O = C @ W2e.t()                             # [Tk, H]

            w_tok = weights.index_select(0, token_idx)[:, ge]  # [Tk]
            out.index_add_(0, token_idx, O * w_tok.unsqueeze(1))

    # -----------------------------------------------------------------
    # 3) Cast to bfloat16 and restore original device
    # -----------------------------------------------------------------
    result = out.to(torch.bfloat16)
    if result.device != orig_devices[2]:      # hidden_states original device
        result = result.to(orig_devices[2])
    return result


# -------------------------------------------------------------
# Export
# -------------------------------------------------------------
__all__ = ["run"]