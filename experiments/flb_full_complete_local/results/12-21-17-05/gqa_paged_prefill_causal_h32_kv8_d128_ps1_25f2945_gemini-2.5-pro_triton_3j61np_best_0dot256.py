# ==============================================================
# gqa_paged_prefill_causal_h32_kv8_d128_ps1 – optimized version
# ==============================================================

import math
import inspect
import torch
import triton
import triton.language as tl


# ----------------------------------------------------------------------
# 1️⃣  Fused‑heads Triton kernel (V‑load → dot product)
# ----------------------------------------------------------------------
@triton.autotune(
    configs=[
        # baseline configs
        triton.Config({'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=2),

        # more aggressive configs (larger BLOCK_N, more warps)
        triton.Config({'BLOCK_N': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 512}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 1024}, num_warps=8, num_stages=4),   # new option
    ],
    key=['total_q'],                # autotuning key (total_q known at launch time)
)
@triton.jit
def _gqa_paged_prefill_causal_kernel_fused(
    # ------------------------------------------------------------------
    # 0️⃣  Pointers
    # ------------------------------------------------------------------
    Q, K_cache, V_cache,
    QO_indptr, KV_indptr, KV_indices,
    Q_seq_idx_map, sm_scale,
    Output, LSE,
    # ------------------------------------------------------------------
    # 1️⃣  Strides
    # ------------------------------------------------------------------
    stride_q_total, stride_q_head, stride_q_dim,
    stride_k_page, stride_k_ps, stride_k_head, stride_k_dim,
    stride_v_page, stride_v_ps, stride_v_head, stride_v_dim,
    # ------------------------------------------------------------------
    # 2️⃣  Meta‑data
    # ------------------------------------------------------------------
    total_q,
    num_qo_heads,
    num_kv_heads,
    # ------------------------------------------------------------------
    # 3️⃣  Compile‑time constants
    # ------------------------------------------------------------------
    GQA_RATIO: tl.constexpr,          # = num_qo_heads // num_kv_heads
    HEAD_DIM: tl.constexpr,           # 128
    BLOCK_N: tl.constexpr,
    PAGE_SIZE: tl.constexpr,          # 1 (hard‑coded)
):
    """
    One program = one (global query token, KV‑head) pair.
    Grid = (total_q, num_kv_heads)
    Inside the program we loop over the GQA_RATIO query heads that share the same KV‑head.
    """

    # -------------------------------------------------
    # 0️⃣ Identify token & KV‑head
    # -------------------------------------------------
    global_q_idx = tl.program_id(0)        # [0, total_q)
    kv_head_idx   = tl.program_id(1)       # [0, num_kv_heads)

    # -------------------------------------------------
    # 1️⃣ Locate the owning sequence (pre‑computed map)
    # -------------------------------------------------
    seq_idx = tl.load(Q_seq_idx_map + global_q_idx)          # int32
    q_start = tl.load(QO_indptr + seq_idx)                  # int32
    q_end   = tl.load(QO_indptr + seq_idx + 1)              # int32
    kv_start = tl.load(KV_indptr + seq_idx)                # int32
    kv_end   = tl.load(KV_indptr + seq_idx + 1)            # int32

    # -------------------------------------------------
    # 2️⃣ Causal limits
    # -------------------------------------------------
    num_q_tokens = q_end - q_start
    num_kv_tokens = kv_end - kv_start
    q_idx_local = global_q_idx - q_start                # position inside the sequence
    delta = num_kv_tokens - num_q_tokens
    max_kv_idx_for_q = q_idx_local + delta + 1          # inclusive count of KV entries this query can attend to

    # ------------------------------------------------------------------
    # 3️⃣ Edge case: no keys to attend to → store zero / -inf and exit
    # ------------------------------------------------------------------
    if max_kv_idx_for_q <= 0:
        head_base = kv_head_idx * GQA_RATIO
        offs_h = tl.arange(0, GQA_RATIO)
        qo_head_idxs = head_base + offs_h                     # [GQA_RATIO]

        offs_d = tl.arange(0, HEAD_DIM)
        out_ptr = Output + global_q_idx * stride_q_total \
                         + qo_head_idxs[:, None] * stride_q_head \
                         + offs_d[None, :]
        tl.store(out_ptr, tl.zeros([GQA_RATIO, HEAD_DIM], dtype=tl.bfloat16))
        lse_ptr = LSE + global_q_idx * num_qo_heads + qo_head_idxs
        tl.store(lse_ptr, -float("inf"))
        return

    # -------------------------------------------------
    # 4️⃣ Load Q for ALL heads that share this KV‑head
    # -------------------------------------------------
    head_base = kv_head_idx * GQA_RATIO
    offs_h = tl.arange(0, GQA_RATIO)                 # [0 .. GQA_RATIO-1]
    qo_head_idxs = head_base + offs_h                # absolute query‑head indices, shape [GQA_RATIO]

    offs_d = tl.arange(0, HEAD_DIM)
    q_ptr = Q + global_q_idx * stride_q_total \
              + qo_head_idxs[:, None] * stride_q_head \
              + offs_d[None, :]
    q = tl.load(q_ptr, mask=True).to(tl.float32)    # [GQA_RATIO, HEAD_DIM]

    # -------------------------------------------------
    # 5️⃣ Allocate per‑head accumulators
    # -------------------------------------------------
    acc = tl.zeros([GQA_RATIO, HEAD_DIM], dtype=tl.float32)   # weighted sum of V
    m_i = tl.full([GQA_RATIO], -float("inf"), dtype=tl.float32)   # running max per head
    l_i = tl.zeros([GQA_RATIO], dtype=tl.float32)                # running exp‑sum per head

    # -------------------------------------------------
    # 6️⃣ Main KV‑loop (block‑wise)
    # -------------------------------------------------
    for n_offset in range(0, max_kv_idx_for_q, BLOCK_N):
        # ---- offsets & validity mask for the block ----
        offs_n = n_offset + tl.arange(0, BLOCK_N)                # [BLOCK_N]
        kv_idx = kv_start + offs_n                               # absolute KV indices
        valid = offs_n < max_kv_idx_for_q                         # respects causal limit
        in_range = kv_idx < kv_end                               # safety (should be true)

        # ---- page ids (gather) ----
        page_ids = tl.load(KV_indices + kv_idx,
                           mask=valid & in_range,
                           other=0)                            # [BLOCK_N]

        # ---- load K block (shared for all heads) ----
        k_ptr = K_cache + (page_ids[:, None] * stride_k_page +
                           kv_head_idx * stride_k_head +
                           offs_d[None, :])
        k = tl.load(k_ptr,
                    mask=valid[:, None] & in_range[:, None],
                    other=0.0).to(tl.float32)                 # [BLOCK_N, HEAD_DIM]

        # ---- compute dot‑products Q·Kᵀ for every head (new, fused version) ----
        # q : [GQA_RATIO, HEAD_DIM] , k : [BLOCK_N, HEAD_DIM] → need Kᵀ : [HEAD_DIM, BLOCK_N]
        s = tl.dot(q, tl.trans(k))                               # [GQA_RATIO, BLOCK_N]

        # ---- scale and mask ----
        s = s * sm_scale
        s = tl.where(valid, s, -float("inf"))

        # ---- online soft‑max per head ----
        m_new = tl.maximum(m_i, tl.max(s, axis=1))                # [GQA_RATIO]
        p = tl.exp(s - m_new[:, None])                           # [GQA_RATIO, BLOCK_N]
        l_new = tl.exp(m_i - m_new) * l_i + tl.sum(p, axis=1)    # [GQA_RATIO]

        # ---- rescale previous accumulator before adding new contribution ----
        acc = acc * tl.exp(m_i - m_new)[:, None]

        # ---- load V block (contiguous in BLOCK_N dimension) ----
        v_ptr = V_cache + (page_ids[:, None] * stride_v_page +
                           kv_head_idx * stride_v_head +
                           offs_d[None, :])
        v = tl.load(v_ptr,
                    mask=valid[:, None] & in_range[:, None],
                    other=0.0).to(tl.float32)                 # [BLOCK_N, HEAD_DIM]

        # ---- accumulate weighted V via a batched matrix‑multiply ----
        # p: [GQA_RATIO, BLOCK_N] , v: [BLOCK_N, HEAD_DIM]
        # result: [GQA_RATIO, HEAD_DIM]
        acc_update = tl.dot(p, v)                                # fp32
        acc = acc + acc_update

        # ---- commit new running statistics for next iteration ----
        m_i = m_new
        l_i = l_new

    # -------------------------------------------------
    # 7️⃣ Finalise per‑head results
    # -------------------------------------------------
    o = acc / l_i[:, None]                 # [GQA_RATIO, HEAD_DIM]  (fp32)
    lse = m_i + tl.log(l_i)                # natural‑log
    lse = lse * 1.4426950408889634         # convert to log₂

    # -------------------------------------------------
    # 8️⃣ Store results
    # -------------------------------------------------
    out_ptr = Output + global_q_idx * stride_q_total \
                       + qo_head_idxs[:, None] * stride_q_head \
                       + offs_d[None, :]
    tl.store(out_ptr, o.to(tl.bfloat16))

    lse_ptr = LSE + global_q_idx * num_qo_heads + qo_head_idxs
    tl.store(lse_ptr, lse)


# ----------------------------------------------------------------------
# 2️⃣  Host‑side wrapper (API unchanged)
# ----------------------------------------------------------------------
def gqa_paged_prefill_causal_h32_kv8_d128_ps1(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare strides, sequence‑map and launch the fused‑heads Triton kernel.
    """
    # ---- validation -------------------------------------------------
    assert q.dim() == 3, "q must be a 3‑D tensor"
    assert k_cache.dim() == 4, "k_cache must be a 4‑D tensor"
    assert v_cache.dim() == 4, "v_cache must be a 4‑D tensor"
    assert q.dtype == torch.bfloat16
    assert k_cache.dtype == torch.bfloat16
    assert v_cache.dtype == torch.bfloat16
    assert qo_indptr.dtype == torch.int32
    assert kv_indptr.dtype == torch.int32
    assert kv_indices.dtype == torch.int32

    total_q, num_qo_heads, head_dim = q.shape
    num_pages, page_size, num_kv_heads, _ = k_cache.shape

    # ---- constants -------------------------------------------------
    assert num_qo_heads == 32
    assert num_kv_heads == 8
    assert head_dim == 128
    assert page_size == 1

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # ---- allocate outputs -------------------------------------------
    output = torch.empty_like(q)                                 # bf16
    lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q.device)

    # ---- pre‑compute map: global query idx → sequence idx ----------
    batch_size = qo_indptr.numel() - 1
    q_seq_len = qo_indptr[1:] - qo_indptr[:-1]                    # lengths per sequence
    q_seq_idx_map = torch.arange(batch_size, device=q.device, dtype=torch.int32) \
                        .repeat_interleave(q_seq_len)

    # ---- launch grid ------------------------------------------------
    grid = (total_q, num_kv_heads)   # one program per (query token, KV‑head)

    _gqa_paged_prefill_causal_kernel_fused[grid](
        # tensors
        q, k_cache, v_cache,
        qo_indptr, kv_indptr, kv_indices,
        q_seq_idx_map, sm_scale,
        output, lse,
        # strides
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        # meta‑data
        total_q,
        num_qo_heads,
        num_kv_heads,
        # compile‑time constants
        GQA_RATIO=num_qo_heads // num_kv_heads,   # 4
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
    )
    return output, lse


# ----------------------------------------------------------------------
# 3️⃣  Public entry point `run` – mirrors the reference implementation
# ----------------------------------------------------------------------
def run(*args, **kwargs):
    """
    Validates arguments, moves everything to CUDA, calls the Triton
    implementation and returns results on the original device of ``q``.
    """
    # -------------------------------------------------
    # 1️⃣ Bind arguments to the core signature
    # -------------------------------------------------
    sig = inspect.signature(gqa_paged_prefill_causal_h32_kv8_d128_ps1)
    try:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
    except TypeError as e:
        raise TypeError(f"Argument binding error: {e}") from e

    # -------------------------------------------------
    # 2️⃣ Extract tensors & optional scalar
    # -------------------------------------------------
    q = bound.arguments["q"]
    k_cache = bound.arguments["k_cache"]
    v_cache = bound.arguments["v_cache"]
    qo_indptr = bound.arguments["qo_indptr"]
    kv_indptr = bound.arguments["kv_indptr"]
    kv_indices = bound.arguments["kv_indices"]
    sm_scale = bound.arguments.get("sm_scale", None)

    # -------------------------------------------------
    # 3️⃣ Ensure a CUDA device is available
    # -------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError("Triton kernels require a CUDA device.")

    # -------------------------------------------------
    # 4️⃣ Move everything to the (first) CUDA tensor's device
    # -------------------------------------------------
    target_device = next(
        (t.device for t in (q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices)
         if isinstance(t, torch.Tensor) and t.is_cuda),
        torch.device("cuda"),
    )
    q = q.to(target_device)
    k_cache = k_cache.to(target_device)
    v_cache = v_cache.to(target_device)
    qo_indptr = qo_indptr.to(target_device)
    kv_indptr = kv_indptr.to(target_device)
    kv_indices = kv_indices.to(target_device)

    # -------------------------------------------------
    # 5️⃣ Call the optimized Triton implementation
    # -------------------------------------------------
    output, lse = gqa_paged_prefill_causal_h32_kv8_d128_ps1(
        q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale
    )

    # -------------------------------------------------
    # 6️⃣ Return results on the original device of ``q``
    # -------------------------------------------------
    orig_device = bound.arguments["q"].device
    return output.to(orig_device), lse.to(orig_device)