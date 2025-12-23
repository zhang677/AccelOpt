# ------------------------------------------------------------
# Triton kernel – one program per (batch, KV‑head)
# ------------------------------------------------------------
import math
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64},  num_warps=4,  num_stages=2),
        triton.Config({"BLOCK_M": 64},  num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_M":128},  num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_M":128},  num_warps=8,  num_stages=3),
        triton.Config({"BLOCK_M":256},  num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_M":256},  num_warps=16, num_stages=2),
    ],
    # “max_seq_len” is the longest KV‑sequence in the batch;
    # it is passed from the host as a launch‑time key.
    key=["batch_size", "max_seq_len"],
)
@triton.jit
def gqa_paged_decode_h32_kv4_d128_ps1_kernel(
    # --------------------- pointers ---------------------
    q_ptr,               # *bf16  [B, Hq, D]
    k_cache_ptr,         # *bf16  [P, 1, Hk, D]
    v_cache_ptr,         # *bf16  [P, 1, Hk, D]
    kv_indptr_ptr,       # *i32   [B+1]
    kv_indices_ptr,      # *i32   [N]
    sm_scale,            # f32 scalar
    output_ptr,          # *bf16  [B, Hq, D]
    lse_ptr,             # *f32   [B, Hq]
    # --------------------- sizes -----------------------
    batch_size,          # i32
    max_seq_len,         # i32  (longest KV length in the batch)
    # --------------------- strides ----------------------
    stride_q_b, stride_q_h, stride_q_d,
    stride_k_p, stride_k_s, stride_k_h, stride_k_d,
    stride_v_p, stride_v_s, stride_v_h, stride_v_d,
    stride_o_b, stride_o_h, stride_o_d,
    stride_lse_b, stride_lse_h,
    # --------------- compile‑time constants -------------
    BLOCK_M: tl.constexpr,          # token block size (tuned)
    D_HEAD: tl.constexpr = 128,    # head dimension (fixed)
    GQA_RATIO: tl.constexpr = 8,   # Qo heads per KV head (fixed)
    NUM_KV_HEADS: tl.constexpr = 4 # number of KV heads (fixed)
):
    """
    One program processes a *single* (batch, KV‑head) tuple.
    All `GQA_RATIO` (=8) Qo‑heads that share the KV‑head are handled
    simultaneously (vectorised over the first dimension).
    Heavy matmuls stay in BF16 so that Tensor‑Core WMMA is used.
    """
    pid = tl.program_id(0)

    # ----------------------------------------------------------------
    # Decode (batch, KV‑head) from the linear program id
    # ----------------------------------------------------------------
    b   = pid // NUM_KV_HEADS          # batch index
    kv_h = pid %  NUM_KV_HEADS          # KV‑head index

    if b >= batch_size:
        return

    # ----------------------------------------------------------------
    # KV‑range for this batch element
    # ----------------------------------------------------------------
    page_start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    page_end   = tl.load(kv_indptr_ptr + (b + 1)).to(tl.int32)
    seq_len = page_end - page_start          # number of tokens for this sequence

    # ----------------------------------------------------------------
    # Early‑exit for empty sequences (store zeros & -inf LSE for the
    # current group of Qo‑heads)
    # ----------------------------------------------------------------
    neg_inf = tl.full((), -float("inf"), dtype=tl.float32)
    if seq_len <= 0:
        d_off = tl.arange(0, D_HEAD)                     # [D]
        h_off = kv_h * GQA_RATIO + tl.arange(0, GQA_RATIO)  # [G]

        # output = 0, lse = -inf for the heads in this KV‑head
        out_ptrs = (output_ptr
                    + b * stride_o_b
                    + h_off[:, None] * stride_o_h
                    + d_off[None, :] * stride_o_d)          # (G, D)
        tl.store(out_ptrs, tl.zeros((GQA_RATIO, D_HEAD), dtype=tl.bfloat16))

        lse_ptrs = (lse_ptr
                    + b * stride_lse_b
                    + h_off * stride_lse_h)                 # (G)
        tl.store(lse_ptrs, tl.full((GQA_RATIO,), neg_inf, dtype=tl.float32))
        return

    # ----------------------------------------------------------------
    # Load the Q‑vectors belonging to this KV‑head (shape GQA_RATIO × D)
    # ----------------------------------------------------------------
    d_off = tl.arange(0, D_HEAD)                     # [D]
    h_off = kv_h * GQA_RATIO + tl.arange(0, GQA_RATIO)  # [G]
    q_ptrs = (q_ptr
              + b * stride_q_b
              + h_off[:, None] * stride_q_h
              + d_off[None, :] * stride_q_d)          # (G, D)
    q_mat = tl.load(q_ptrs)                          # (G, D)  bf16   ← keep BF16

    # ----------------------------------------------------------------
    # Streaming‑softmax state per Qo‑head (stays FP32)
    # ----------------------------------------------------------------
    m_i = tl.full((GQA_RATIO,), neg_inf, dtype=tl.float32)   # (G)
    l_i = tl.zeros((GQA_RATIO,), dtype=tl.float32)          # (G)
    acc = tl.zeros((GQA_RATIO, D_HEAD), dtype=tl.float32)   # (G, D)

    # ----------------------------------------------------------------
    # Token‑block loop (K/V loaded once per block, reused for all heads)
    # ----------------------------------------------------------------
    token_off = tl.arange(0, BLOCK_M)
    pos = tl.zeros((), dtype=tl.int32)

    while pos < seq_len:
        cur = pos + token_off                     # candidate token ids
        mask = cur < seq_len                       # (M) bool

        # --------------------------------------------------------
        # Load page ids (one per token, because page_size == 1)
        # --------------------------------------------------------
        page_ids = tl.load(
            kv_indices_ptr + page_start + cur,
            mask=mask,
            other=0
        ).to(tl.int32)                           # (M)

        # --------------------------------------------------------
        # Load K block for the current KV‑head (shared for all heads)
        # --------------------------------------------------------
        k_ptrs = (k_cache_ptr
                  + page_ids[:, None] * stride_k_p
                  + 0 * stride_k_s
                  + kv_h * stride_k_h
                  + d_off[None, :] * stride_k_d)   # (M, D)
        k_block = tl.load(k_ptrs,
                         mask=mask[:, None],
                         other=0)                 # (M, D) bf16

        # --------------------------------------------------------
        # Compute logits = Q · Kᵀ   (BF16 matmul → Tensor‑Core)
        # --------------------------------------------------------
        logits_bf16 = tl.dot(q_mat, tl.trans(k_block)) * sm_scale   # (G, M) bf16
        logits = logits_bf16.to(tl.float32)                         # cast once for soft‑max

        # mask padding positions
        logits = tl.where(mask[None, :], logits,
                          tl.full((), -float("inf"), tl.float32))

        # --------------------------------------------------------
        # Streaming‑softmax update (still FP32)
        # --------------------------------------------------------
        m_curr = tl.max(logits, axis=1)                # (G)
        m_new  = tl.maximum(m_i, m_curr)               # (G)

        p = tl.exp(logits - m_new[:, None])            # (G, M) FP32
        l_part = tl.sum(p, axis=1)                     # (G)

        # update normaliser
        l_i = l_i * tl.exp(m_i - m_new) + l_part
        # shift accumulator with the same max‑shift
        acc = acc * tl.exp(m_i - m_new)[:, None]

        # --------------------------------------------------------
        # Load V block (shared for all heads) – keep BF16
        # --------------------------------------------------------
        v_ptrs = (v_cache_ptr
                  + page_ids[:, None] * stride_v_p
                  + 0 * stride_v_s
                  + kv_h * stride_v_h
                  + d_off[None, :] * stride_v_d)   # (M, D)
        v_block = tl.load(v_ptrs,
                         mask=mask[:, None],
                         other=0)                 # (M, D) bf16

        # --------------------------------------------------------
        # Weighted sum: p (FP32) * V (BF16)
        # Cast p to BF16 so tl.dot can use Tensor‑Core
        # --------------------------------------------------------
        p_bf16 = p.to(tl.bfloat16)                     # (G, M) bf16
        weighted_bf16 = tl.dot(p_bf16, v_block)        # (G, D) bf16
        acc += weighted_bf16.to(tl.float32)            # accumulate in FP32

        # --------------------------------------------------------
        # Prepare for next block
        # --------------------------------------------------------
        m_i = m_new
        pos += BLOCK_M

    # ------------------------------------------------------------
    # Finalise each Qo‑head in the group
    # ------------------------------------------------------------
    nonempty = l_i > 0.0
    out = tl.where(
        nonempty[:, None],
        acc / l_i[:, None],
        tl.zeros((GQA_RATIO, D_HEAD), dtype=tl.float32)
    )

    # Store outputs (shape GQA_RATIO × D) back as BF16
    out_ptrs = (output_ptr
                + b * stride_o_b
                + h_off[:, None] * stride_o_h
                + d_off[None, :] * stride_o_d)            # (G, D)
    tl.store(out_ptrs, out.to(tl.bfloat16))

    # 2‑base log‑sum‑exp per Qo‑head
    inv_ln2 = 1.4426950408889634      # 1 / ln(2)
    lse_val = tl.where(
        nonempty,
        (tl.log(l_i) + m_i) * inv_ln2,
        tl.full((GQA_RATIO,), neg_inf, dtype=tl.float32)
    )
    lse_ptrs = (lse_ptr
                + b * stride_lse_b
                + h_off * stride_lse_h)               # (G)
    tl.store(lse_ptrs, lse_val)


# ------------------------------------------------------------
# Helper – ensure tensor is on the correct CUDA device
# ------------------------------------------------------------
def _ensure_cuda(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move tensor to *device* (or copy if on CPU)."""
    if t.device.type == "cuda":
        return t if t.device == device else t.to(device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available – Triton kernels need a GPU.")
    return t.to(device, non_blocking=True)


# ------------------------------------------------------------
# Optimised run() – launches one program per (batch, KV‑head)
# ------------------------------------------------------------
def run(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float | torch.Tensor | None = None,
):
    """
    Triton‑accelerated GQA paged decode with BF16 Tensor‑Core matmuls.
    The API and semantics are identical to the reference implementation.
    """
    # ----------------------------------------------------------------
    # Fixed specification constants
    # ----------------------------------------------------------------
    HEAD_DIM = 128
    NUM_QO_HEADS = 32
    NUM_KV_HEADS = 4
    PAGE_SIZE = 1
    GQA_RATIO = NUM_QO_HEADS // NUM_KV_HEADS          # = 8

    # ----------------------------------------------------------------
    # Default softmax scale
    # ----------------------------------------------------------------
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    # Normalise ``sm_scale`` to a Python float (required for kernel arg)
    if isinstance(sm_scale, (float, int)):
        sm_scale_val = float(sm_scale)
    elif isinstance(sm_scale, torch.Tensor):
        if sm_scale.numel() != 1:
            raise ValueError("sm_scale must be a scalar.")
        sm_scale_val = float(sm_scale.item())
    else:
        raise TypeError("sm_scale must be float, int or a 0‑dim torch.Tensor")

    # ----------------------------------------------------------------
    # Basic sanity checks (mirrors the reference implementation)
    # ----------------------------------------------------------------
    if q.ndim != 3:
        raise ValueError("q must be of shape [B, Hq, D]")
    batch_size, num_qo_heads, head_dim = q.shape
    if (num_qo_heads != NUM_QO_HEADS) or (head_dim != HEAD_DIM):
        raise AssertionError("q must have shape [batch, 32, 128]")

    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError("k_cache / v_cache must be 4‑D")
    if k_cache.shape != v_cache.shape:
        raise AssertionError("k_cache and v_cache must have identical shapes")
    num_pages, page_size, num_kv_heads, head_dim_k = k_cache.shape
    if (num_kv_heads != NUM_KV_HEADS) or (head_dim_k != HEAD_DIM) or (page_size != PAGE_SIZE):
        raise AssertionError("k_cache/v_cache shape mismatch with spec")

    if kv_indptr.ndim != 1 or kv_indices.ndim != 1:
        raise ValueError("kv_indptr / kv_indices must be 1‑D")
    len_indptr = kv_indptr.shape[0]
    if len_indptr != batch_size + 1:
        raise AssertionError("len_indptr must be batch_size + 1")
    if int(kv_indptr[-1].item()) != kv_indices.shape[0]:
        raise AssertionError("kv_indices length inconsistent with kv_indptr[-1]")

    # dtype checks
    if q.dtype != torch.bfloat16:
        raise TypeError("q must be bfloat16")
    if k_cache.dtype != torch.bfloat16 or v_cache.dtype != torch.bfloat16:
        raise TypeError("k_cache and v_cache must be bfloat16")
    if kv_indptr.dtype != torch.int32 or kv_indices.dtype != torch.int32:
        raise TypeError("kv_indptr / kv_indices must be int32")

    # ----------------------------------------------------------------
    # Pick a CUDA device (the first tensor that already lives on GPU wins)
    # ----------------------------------------------------------------
    work_device = (
        q.device if q.is_cuda else
        k_cache.device if k_cache.is_cuda else
        v_cache.device if v_cache.is_cuda else
        kv_indptr.device if kv_indptr.is_cuda else
        kv_indices.device if kv_indices.is_cuda else
        torch.device("cuda")
    )

    # ----------------------------------------------------------------
    # Move everything to the working device and make them contiguous
    # ----------------------------------------------------------------
    q_dev = _ensure_cuda(q.contiguous(), work_device)
    k_dev = _ensure_cuda(k_cache.contiguous(), work_device)
    v_dev = _ensure_cuda(v_cache.contiguous(), work_device)
    indptr_dev = _ensure_cuda(kv_indptr.contiguous(), work_device)
    indices_dev = _ensure_cuda(kv_indices.contiguous(), work_device)

    # ----------------------------------------------------------------
    # Allocate output tensors on the working device
    # ----------------------------------------------------------------
    output_dev = torch.empty(
        (batch_size, NUM_QO_HEADS, HEAD_DIM),
        dtype=torch.bfloat16,
        device=work_device,
    )
    lse_dev = torch.empty(
        (batch_size, NUM_QO_HEADS),
        dtype=torch.float32,
        device=work_device,
    )

    # ----------------------------------------------------------------
    # Compute the maximum sequence length – needed as a tuning key
    # ----------------------------------------------------------------
    indptr_cpu = indptr_dev.cpu()
    seq_lens = indptr_cpu[1:] - indptr_cpu[:-1]
    max_seq_len = int(seq_lens.max().item())

    # ----------------------------------------------------------------
    # Kernel launch configuration
    # ----------------------------------------------------------------
    # One program per (batch, KV‑head)
    grid = (batch_size * NUM_KV_HEADS,)

    gqa_paged_decode_h32_kv4_d128_ps1_kernel[grid](
        # pointers
        q_dev,
        k_dev,
        v_dev,
        indptr_dev,
        indices_dev,
        sm_scale_val,
        output_dev,
        lse_dev,
        # sizes / autotune key
        batch_size,
        max_seq_len,
        # strides (passed in the same order as kernel signature)
        q_dev.stride(0), q_dev.stride(1), q_dev.stride(2),
        k_dev.stride(0), k_dev.stride(1), k_dev.stride(2), k_dev.stride(3),
        v_dev.stride(0), v_dev.stride(1), v_dev.stride(2), v_dev.stride(3),
        output_dev.stride(0), output_dev.stride(1), output_dev.stride(2),
        lse_dev.stride(0), lse_dev.stride(1),
        # compile‑time constants
        D_HEAD=HEAD_DIM,
        GQA_RATIO=GQA_RATIO,
        NUM_KV_HEADS=NUM_KV_HEADS,
    )

    # ----------------------------------------------------------------
    # Return results on the original device of ``q``
    # ----------------------------------------------------------------
    if work_device != q.device:
        output = output_dev.to(q.device)
        lse = lse_dev.to(q.device)
    else:
        output = output_dev
        lse = lse_dev

    return output, lse