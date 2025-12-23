import math
import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_kernel(
    QN,                     # (B, H, 512)       bf16
    QP,                     # (B, H, 64)        bf16
    KC,                     # (P, 512)          bf16
    KP,                     # (P, 64)           bf16
    KV_INDICES,             # (N)               int32
    KV_INDPTR,              # (B + 1)           int32
    SM_SCALE,               # scalar            fp32
    OUT,                    # (B, H, 512)       bf16
    LSE,                    # (B, H)            fp32
    B: tl.constexpr,
    H: tl.constexpr,
    D_CKV: tl.constexpr,
    D_KPE: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
):
    """
    One Triton program computes a single (batch, head) pair.
    page_size == 1, num_qo_heads == 16, D_CKV == 512, D_KPE == 64
    """

    pid = tl.program_id(axis=0)
    b = pid // H                      # batch index
    h = pid % H                       # head  index

    # ------------------- offsets ------------------- #
    offs_ckv = tl.arange(0, D_CKV)            # (512,)
    offs_kpe = tl.arange(0, D_KPE)            # (64,)
    offs_t   = tl.arange(0, BLOCK_TOK)        # (T,)

    # ------------------- KV range ------------------ #
    kv_beg = tl.load(KV_INDPTR + b)
    kv_end = tl.load(KV_INDPTR + b + 1)
    kv_len = kv_end - kv_beg                  # scalar int32

    # Pointers to output locations
    ptr_out = OUT + (b * H + h) * D_CKV + offs_ckv
    ptr_lse = LSE + b * H + h

    # If there is no KV data, write zeros / -inf and exit.
    if kv_len <= 0:
        tl.store(ptr_out, tl.zeros([D_CKV], dtype=tl.bfloat16))
        tl.store(ptr_lse, -float("inf"))
        return

    # ------------------- load queries -------------- #
    qn_ptr = QN + (b * H + h) * D_CKV + offs_ckv
    qp_ptr = QP + (b * H + h) * D_KPE + offs_kpe
    qn = tl.load(qn_ptr).to(tl.float32)        # (512,)
    qp = tl.load(qp_ptr).to(tl.float32)        # (64,)

    # ------------------- accumulators -------------- #
    s_sum = tl.zeros([], dtype=tl.float32)     # scalar
    w_sum = tl.zeros([D_CKV], dtype=tl.float32)

    tok_start = tl.zeros([], dtype=tl.int32)   # current token pointer

    while tok_start < kv_len:
        remaining = kv_len - tok_start
        block_n = tl.where(remaining < BLOCK_TOK, remaining, BLOCK_TOK)  # scalar int32
        mask_t = offs_t < block_n                                        # (T,)

        # --- gather token indices -------------------------------------- #
        idx_ptr  = KV_INDICES + kv_beg + tok_start + offs_t
        tok_idx  = tl.load(idx_ptr, mask=mask_t, other=0)                # (T,)

        # --- gather KC, KP --------------------------------------------- #
        kc_ptr = KC + tok_idx[:, None] * D_CKV + offs_ckv[None, :]
        kp_ptr = KP + tok_idx[:, None] * D_KPE + offs_kpe[None, :]

        kc_blk = tl.load(kc_ptr, mask=mask_t[:, None], other=0).to(tl.float32)  # (T,512)
        kp_blk = tl.load(kp_ptr, mask=mask_t[:, None], other=0).to(tl.float32)  # (T,64)

        # --- compute logits -------------------------------------------- #
        l_ckv  = tl.sum(kc_blk * qn[None, :], axis=1)          # (T,)
        l_kpe  = tl.sum(kp_blk * qp[None, :], axis=1)          # (T,)
        logits = (l_ckv + l_kpe) * SM_SCALE                   # (T,)

        exp_logits = tl.exp(logits)
        exp_logits = tl.where(mask_t, exp_logits, 0.0)

        # --- accumulate ------------------------------------------------- #
        s_sum += tl.sum(exp_logits, axis=0)                               # scalar
        w_sum += tl.sum(exp_logits[:, None] * kc_blk, axis=0)             # (512,)

        tok_start += BLOCK_TOK

    # ------------------- write back ------------------------------------ #
    inv_ln2 = 1.4426950408889634  # 1 / ln(2)
    out_vec = w_sum / s_sum
    log_s   = tl.log(s_sum) * inv_ln2

    tl.store(ptr_out, out_vec.to(tl.bfloat16))
    tl.store(ptr_lse, log_s)


def run(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
):
    """
    Optimized paged-decode kernel for (H=16, D_CKV=512, D_KPE=64, page_size=1).

    Inputs:
        q_nope     : (B, 16, 512)  bfloat16
        q_pe       : (B, 16, 64)   bfloat16
        ckv_cache  : (P, 1, 512)   bfloat16
        kpe_cache  : (P, 1, 64)    bfloat16
        kv_indptr  : (B + 1)       int32
        kv_indices : (N)           int32
        sm_scale   : float (fp32)

    Returns:
        dict(output=(B,16,512) bf16, lse=(B,16) fp32)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required to run Triton kernels.")

    # ------------------- validation ------------------- #
    assert q_nope.dtype == torch.bfloat16 and q_pe.dtype == torch.bfloat16
    B, H, D_CKV = q_nope.shape
    assert H == 16 and D_CKV == 512
    assert q_pe.shape == (B, 16, 64)
    assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1            # page_size = 1
    assert kv_indptr.shape[0] == B + 1
    assert kv_indices.shape[0] == kv_indptr[-1].item()

    # ---------------- device handling ----------------- #
    orig_device = q_nope.device
    cuda_dev = torch.cuda.current_device()

    def _to_cuda(t: torch.Tensor):
        return t.to(device=cuda_dev, non_blocking=True) if not t.is_cuda else t

    q_nope_d = _to_cuda(q_nope)
    q_pe_d   = _to_cuda(q_pe)
    kc_d     = _to_cuda(ckv_cache.squeeze(1))
    kp_d     = _to_cuda(kpe_cache.squeeze(1))
    indptr_d = _to_cuda(kv_indptr)
    indices_d= _to_cuda(kv_indices)

    # ---------------- output buffers ------------------ #
    out_d = torch.empty((B, H, 512), dtype=torch.bfloat16, device=cuda_dev)
    lse_d = torch.empty((B, H), dtype=torch.float32, device=cuda_dev)

    # ---------------- kernel launch ------------------- #
    BLOCK_TOK = 128
    grid = (B * H,)

    _paged_decode_kernel[grid](
        q_nope_d,
        q_pe_d,
        kc_d,
        kp_d,
        indices_d,
        indptr_d,
        float(sm_scale),
        out_d,
        lse_d,
        B=B,
        H=H,
        D_CKV=512,
        D_KPE=64,
        BLOCK_TOK=BLOCK_TOK,
        num_warps=8,
        num_stages=4,
    )

    # --------------- move outputs back --------------- #
    if orig_device.type == "cpu":
        out_d = out_d.cpu()
        lse_d = lse_d.cpu()
    elif orig_device != out_d.device:
        out_d = out_d.to(orig_device)
        lse_d = lse_d.to(orig_device)

    return out_d, lse_d