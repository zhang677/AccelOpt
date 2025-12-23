# --------------------------------------------------------------
# mla_paged_decode_h16_ckv512_kpe64_ps1 – bf16 Tensor‑Core version
# --------------------------------------------------------------
#   * One Triton program per batch element (grid = B)
#   * 16 heads are processed together (H_TILE = 16)
#   * KV vectors are loaded once per token block and reused
#   * bf16 data stays in bf16 up to the large matrix‑vector multiplies
#   * dot‑products are executed on Tensor‑Cores (out_dtype=fp32)
# --------------------------------------------------------------

import math
import torch
import triton
import triton.language as tl


# ------------------------------------------------------------------
# 1️⃣  Autotuned kernel – fused heads, bf16 Tensor‑Core arithmetic
# ------------------------------------------------------------------
@triton.autotune(
    configs=[
        # legacy (single‑head) configs – keep them as fallback
        triton.Config({'BLOCK_TOK':  64, 'H_TILE': 1}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_TOK': 128, 'H_TILE': 1}, num_warps=8,  num_stages=4),
        # full‑head fusion configs (H_TILE = 16)
        triton.Config({'BLOCK_TOK':  64, 'H_TILE': 16}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_TOK': 128, 'H_TILE': 16}, num_warps=16, num_stages=5),
        triton.Config({'BLOCK_TOK': 256, 'H_TILE': 16}, num_warps=16, num_stages=5),
        triton.Config({'BLOCK_TOK': 512, 'H_TILE': 16}, num_warps=32, num_stages=5),
    ],
    key=['B', 'D_CKV', 'D_KPE']   # H is a compile‑time constant (16), H_TILE lives in the config
)
@triton.jit
def _paged_decode_kernel_fused_head(
    QN,                     # (B, 16, 512)      bf16
    QP,                     # (B, 16, 64)       bf16
    KC,                     # (P, 512)          bf16
    KP,                     # (P, 64)           bf16
    KV_INDICES,             # (N)               int32
    KV_INDPTR,              # (B+1)             int32
    SM_SCALE,               # scalar            fp32
    OUT,                    # (B, 16, 512)      bf16
    LSE,                    # (B, 16)           fp32
    #
    B: tl.constexpr,       # batch size
    D_CKV: tl.constexpr,   # 512
    D_KPE: tl.constexpr,   # 64
    BLOCK_TOK: tl.constexpr,
    H_TILE: tl.constexpr,  # number of heads processed together (1 or 16)
):
    """
    One Triton program processes **H_TILE** heads of a single batch element.
    The KV block is loaded once and reused for all heads, dramatically
    reducing global‑memory traffic.
    All heavy arithmetic (QN·KCᵀ, QP·KPᵀ) stays in bf16 and runs on Tensor‑Cores.
    """
    pid = tl.program_id(0)                # one program per batch element
    b = pid                                 # batch index

    H = 16                                   # total heads (constant)
    mask_h = tl.arange(0, H_TILE) < H       # mask for possibly‑partial tile (fallback case)

    # --------------------------------------------------------------
    # 2️⃣  Offsets
    # --------------------------------------------------------------
    offs_ckv = tl.arange(0, D_CKV)          # (512,)
    offs_kpe = tl.arange(0, D_KPE)          # (64,)
    offs_tok = tl.arange(0, BLOCK_TOK)      # (BLOCK_TOK,)

    # --------------------------------------------------------------
    # 3️⃣  Load queries for the whole tile (keep bf16)
    # --------------------------------------------------------------
    qn_ptr = QN + (b * H + tl.arange(0, H_TILE)[:, None]) * D_CKV + offs_ckv[None, :]
    qp_ptr = QP + (b * H + tl.arange(0, H_TILE)[:, None]) * D_KPE + offs_kpe[None, :]

    qn = tl.load(qn_ptr,
                 mask=mask_h[:, None],
                 other=0)                 # (H_TILE, D_CKV) bf16
    qp = tl.load(qp_ptr,
                 mask=mask_h[:, None],
                 other=0)                 # (H_TILE, D_KPE) bf16

    # --------------------------------------------------------------
    # 4️⃣  KV range for this batch element
    # --------------------------------------------------------------
    kv_beg = tl.load(KV_INDPTR + b)          # first token index
    kv_end = tl.load(KV_INDPTR + b + 1)      # exclusive
    kv_len = kv_end - kv_beg

    # --------------------------------------------------------------
    # 5️⃣  Edge case – no KV tokens for this batch element
    # --------------------------------------------------------------
    if kv_len <= 0:
        out_ptr = OUT + (b * H) * D_CKV + tl.arange(0, H_TILE)[:, None] * D_CKV + offs_ckv[None, :]
        lse_ptr = LSE + b * H + tl.arange(0, H_TILE)

        tl.store(out_ptr,
                 tl.zeros([H_TILE, D_CKV], dtype=tl.bfloat16),
                 mask=mask_h[:, None])
        tl.store(lse_ptr,
                 tl.full([H_TILE], -float("inf"), dtype=tl.float32),
                 mask=mask_h)
        return

    # --------------------------------------------------------------
    # 6️⃣  Per‑tile accumulators (one per head)
    # --------------------------------------------------------------
    s_sum = tl.zeros([H_TILE], dtype=tl.float32)                # Σ exp(logits) per head
    w_sum = tl.zeros([H_TILE, D_CKV], dtype=tl.float32)         # Σ exp(logits) * KC per head

    # --------------------------------------------------------------
    # 7️⃣  Token loop – process tokens in blocks of BLOCK_TOK
    # --------------------------------------------------------------
    tok_start = tl.zeros([], dtype=tl.int32)

    while tok_start < kv_len:
        remaining = kv_len - tok_start
        cur_block = tl.where(remaining < BLOCK_TOK, remaining, BLOCK_TOK)
        mask_t = offs_tok < cur_block                               # (BLOCK_TOK,)

        # ---- gather token indices ---------------------------------
        idx_ptr = KV_INDICES + kv_beg + tok_start + offs_tok
        tok_idx = tl.load(idx_ptr, mask=mask_t, other=0)            # (BLOCK_TOK,)

        # ---- shared KV loads (once per block, used by all heads) --
        kc_ptr = KC + tok_idx[:, None] * D_CKV + offs_ckv[None, :]   # (T, 512)
        kp_ptr = KP + tok_idx[:, None] * D_KPE + offs_kpe[None, :]   # (T, 64)

        kc_blk = tl.load(kc_ptr,
                         mask=mask_t[:, None],
                         other=0)                                 # (T, 512) bf16
        kp_blk = tl.load(kp_ptr,
                         mask=mask_t[:, None],
                         other=0)                                 # (T, 64)  bf16

        # ---- batched dot‑products for all heads in the tile ------
        # Tensor‑Core bf16·bf16 → fp32 result
        l_ckv = tl.dot(qn, tl.trans(kc_blk), out_dtype=tl.float32) # (H_TILE, T)
        l_kpe = tl.dot(qp, tl.trans(kp_blk), out_dtype=tl.float32) # (H_TILE, T)

        logits = (l_ckv + l_kpe) * SM_SCALE                        # (H_TILE, T)

        exp_logits = tl.exp(logits)
        # mask out the padded part of the block
        exp_logits = tl.where(mask_t[None, :], exp_logits, 0.0)     # (H_TILE, T)

        # ---- accumulate ------------------------------------------------
        s_sum += tl.sum(exp_logits, axis=1)                        # (H_TILE,)

        # For the weighted sum we need the KV values in fp32
        kc_blk_fp32 = kc_blk.to(tl.float32)

        # (H_TILE, D_CKV) = (H_TILE, T) @ (T, D_CKV)
        w_sum += tl.dot(exp_logits, kc_blk_fp32)                    # (H_TILE, D_CKV)

        # ---- advance ---------------------------------------------------
        tok_start += BLOCK_TOK

    # ------------------------------------------------------------------
    # 8️⃣  Final reduction → output vector and 2‑base LSE
    # ------------------------------------------------------------------
    inv_ln2 = 1.4426950408889634          # 1 / ln(2)
    out_vec = w_sum / s_sum[:, None]      # (H_TILE, D_CKV)
    lse_val = tl.log(s_sum) * inv_ln2     # (H_TILE,)

    # ------------------------------------------------------------------
    # 9️⃣  Store results (only the valid heads of the tile)
    # ------------------------------------------------------------------
    out_ptr = OUT + (b * H) * D_CKV + tl.arange(0, H_TILE)[:, None] * D_CKV + offs_ckv[None, :]
    lse_ptr = LSE + b * H + tl.arange(0, H_TILE)

    tl.store(out_ptr, out_vec.to(tl.bfloat16), mask=mask_h[:, None])
    tl.store(lse_ptr, lse_val, mask=mask_h)


# --------------------------------------------------------------
# Python wrapper – same public API as the reference implementation
# --------------------------------------------------------------
def run(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Optimized paged‑decode kernel with full‑head fusion and bf16 Tensor‑Core dot products.
    Returns
    -------
    output : torch.Tensor  (B, 16, 512)  bf16
    lse    : torch.Tensor  (B, 16)       fp32
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required to run Triton kernels.")

    # -------------------- validation -------------------- #
    assert q_nope.dtype == torch.bfloat16 and q_pe.dtype == torch.bfloat16
    B, H, D_CKV = q_nope.shape
    assert H == 16 and D_CKV == 512
    assert q_pe.shape == (B, 16, 64)
    assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1   # page_size = 1
    assert kv_indptr.shape[0] == B + 1
    assert kv_indices.shape[0] == kv_indptr[-1].item()

    # ---------------- device handling ----------------- #
    orig_device = q_nope.device
    cuda_dev = torch.cuda.current_device()

    def _to_cuda(t: torch.Tensor) -> torch.Tensor:
        return t.to(device=cuda_dev, non_blocking=True) if not t.is_cuda else t

    q_nope_d = _to_cuda(q_nope)                                 # (B,16,512)
    q_pe_d   = _to_cuda(q_pe)                                   # (B,16,64)
    kc_d     = _to_cuda(ckv_cache.squeeze(1))                   # (P,512) bf16
    kp_d     = _to_cuda(kpe_cache.squeeze(1))                   # (P,64)  bf16
    indptr_d = _to_cuda(kv_indptr)                             # (B+1,)  int32
    indices_d= _to_cuda(kv_indices)                            # (N,)    int32

    # ---------------- output buffers ------------------ #
    out_d = torch.empty((B, H, D_CKV), dtype=torch.bfloat16, device=cuda_dev)
    lse_d = torch.empty((B, H),        dtype=torch.float32, device=cuda_dev)

    # ---------------- kernel launch ------------------- #
    grid = (B,)   # one program per batch element

    _paged_decode_kernel_fused_head[grid](
        q_nope_d,
        q_pe_d,
        kc_d,
        kp_d,
        indices_d,
        indptr_d,
        float(sm_scale),
        out_d,
        lse_d,
        # compile‑time constants
        B=B,
        D_CKV=512,
        D_KPE=64,
        # BLOCK_TOK and H_TILE are supplied by the autotuner
    )

    # ---------------- move results back ---------------- #
    if orig_device.type == "cpu":
        out_d = out_d.cpu()
        lse_d = lse_d.cpu()
    elif orig_device != out_d.device:
        out_d = out_d.to(orig_device)
        lse_d = lse_d.to(orig_device)

    return out_d, lse_d