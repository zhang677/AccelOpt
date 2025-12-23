import math
import torch
import triton
import triton.language as tl


@triton.jit
def gqa_paged_decode_h32_kv4_d128_ps1_kernel(
    q_ptr,                         # *bf16 [B, Hq, D]
    k_cache_ptr,                   # *bf16 [P, S=1, Hk=4, D]
    v_cache_ptr,                   # *bf16 [P, S=1, Hk=4, D]
    kv_indptr_ptr,                 # *i32  [B+1]
    kv_indices_ptr,                # *i32  [N]
    sm_scale,                      # f32 scalar
    output_ptr,                    # *bf16 [B, Hq, D]
    lse_ptr,                       # *f32  [B, Hq]
    batch_size,                    # i32
    num_qo_heads,                  # i32 (should be 32)
    # strides for q
    stride_q_b, stride_q_h, stride_q_d,
    # strides for k_cache
    stride_k_p, stride_k_s, stride_k_h, stride_k_d,
    # strides for v_cache
    stride_v_p, stride_v_s, stride_v_h, stride_v_d,
    # strides for output
    stride_o_b, stride_o_h, stride_o_d,
    # strides for lse
    stride_lse_b, stride_lse_h,
    BLOCK_M: tl.constexpr,         # token block size
    D_HEAD: tl.constexpr,          # head dim = 128
    GQA_RATIO: tl.constexpr        # 8
):
    pid = tl.program_id(0)
    # Compute (b, h)
    h = pid % num_qo_heads
    b = pid // num_qo_heads
    if b >= batch_size:
        return

    # Load KV range for this batch
    page_start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    page_end = tl.load(kv_indptr_ptr + (b + 1)).to(tl.int32)
    seq_len = page_end - page_start

    # Early exit for empty sequences
    if seq_len <= 0:
        d_offsets = tl.arange(0, D_HEAD)
        o_ptrs = output_ptr + b * stride_o_b + h * stride_o_h + d_offsets * stride_o_d
        tl.store(o_ptrs, tl.zeros((D_HEAD,), dtype=tl.bfloat16))
        lse_out_ptr = lse_ptr + b * stride_lse_b + h * stride_lse_h
        neg_inf = -float("inf")
        tl.store(lse_out_ptr, tl.full((), neg_inf, dtype=tl.float32))
        return

    # Load Q vector for this (b, h)
    d_offsets = tl.arange(0, D_HEAD)
    q_ptrs = q_ptr + b * stride_q_b + h * stride_q_h + d_offsets * stride_q_d
    q_vec = tl.load(q_ptrs).to(tl.float32)

    # Initialize streaming softmax state
    neg_inf = tl.full((), -float("inf"), dtype=tl.float32)
    m_i = neg_inf
    l_i = tl.zeros((), dtype=tl.float32)
    acc = tl.zeros((D_HEAD,), dtype=tl.float32)

    # Determine KV head from GQA mapping
    kv_head = (h // GQA_RATIO).to(tl.int32)

    # Iterate over tokens in blocks
    pos = tl.zeros((), dtype=tl.int32)
    while pos < seq_len:
        t_offsets = tl.arange(0, BLOCK_M)
        curr = pos + t_offsets
        mask_t = curr < seq_len

        # Gather page ids
        page_ids = tl.load(kv_indices_ptr + page_start + curr, mask=mask_t, other=0).to(tl.int32)

        # K pointers [BLOCK_M, D_HEAD]
        k_ptrs = (
            k_cache_ptr
            + page_ids[:, None] * stride_k_p
            + 0 * stride_k_s
            + kv_head * stride_k_h
            + d_offsets[None, :] * stride_k_d
        )
        k_block = tl.load(k_ptrs, mask=mask_t[:, None], other=0).to(tl.float32)

        # Compute logits for this block: [BLOCK_M]
        logits = tl.sum(k_block * q_vec[None, :], axis=1)
        logits_scaled = logits * sm_scale
        logits_scaled = tl.where(mask_t, logits_scaled, neg_inf)

        # Block-level max
        m_curr = tl.max(logits_scaled, axis=0)
        m_new = tl.maximum(m_i, m_curr)

        # Compute p = exp(logits - m_new)
        p = tl.exp(logits_scaled - m_new)
        # sum of p
        l_part = tl.sum(p, axis=0)
        # Update l_i
        l_i = l_i * tl.exp(m_i - m_new) + l_part

        # V pointers and weighted accumulation
        v_ptrs = (
            v_cache_ptr
            + page_ids[:, None] * stride_v_p
            + 0 * stride_v_s
            + kv_head * stride_v_h
            + d_offsets[None, :] * stride_v_d
        )
        v_block = tl.load(v_ptrs, mask=mask_t[:, None], other=0).to(tl.float32)
        weighted = tl.sum(v_block * p[:, None], axis=0)

        # Update accumulator and max
        acc = acc * tl.exp(m_i - m_new) + weighted
        m_i = m_new

        pos += BLOCK_M

    # Finalize output
    nonempty = l_i > 0.0
    out_vec = tl.where(nonempty, acc / l_i, tl.zeros((D_HEAD,), dtype=tl.float32))
    inv_ln2 = 1.4426950408889634  # 1 / ln(2)
    lse_val = tl.where(nonempty, (tl.log(l_i) + m_i) * inv_ln2, neg_inf)

    # Store output and lse
    o_ptrs = output_ptr + b * stride_o_b + h * stride_o_h + d_offsets * stride_o_d
    tl.store(o_ptrs, out_vec.to(tl.bfloat16))
    lse_out_ptr = lse_ptr + b * stride_lse_b + h * stride_lse_h
    tl.store(lse_out_ptr, lse_val)


def _ensure_cuda(t: torch.Tensor, device: torch.device):
    if t.device.type == "cuda":
        if t.device != device:
            return t.to(device)
        return t
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, but GPU execution is required.")
        return t.to(device, non_blocking=True)


def run(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale=None):
    # Validate and default sm_scale
    HEAD_DIM = 128
    NUM_QO_HEADS = 32
    NUM_KV_HEADS = 4
    PAGE_SIZE = 1
    GQA_RATIO = NUM_QO_HEADS // NUM_KV_HEADS
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(HEAD_DIM)

    # Convert sm_scale to Python float
    if isinstance(sm_scale, (float, int)):
        sm_scale_val = float(sm_scale)
    elif isinstance(sm_scale, torch.Tensor):
        if sm_scale.numel() != 1:
            raise ValueError("sm_scale must be a scalar.")
        sm_scale_val = float(sm_scale.detach().cpu().item())
    else:
        raise TypeError("sm_scale must be a float, int, or 0-dim torch.Tensor")

    # Extract shapes and validate
    if q.ndim != 3:
        raise ValueError("q must have shape [batch_size, num_qo_heads, head_dim]")
    batch_size, num_qo_heads, head_dim = q.shape
    if k_cache.ndim != 4:
        raise ValueError("k_cache must have shape [num_pages, page_size, num_kv_heads, head_dim]")
    if v_cache.ndim != 4:
        raise ValueError("v_cache must have shape [num_pages, page_size, num_kv_heads, head_dim]")

    num_pages, page_size, num_kv_heads, head_dim_k = k_cache.shape
    num_pages_v, page_size_v, num_kv_heads_v, head_dim_v = v_cache.shape

    if num_pages != num_pages_v or page_size != page_size_v or num_kv_heads != num_kv_heads_v or head_dim_k != head_dim_v:
        raise ValueError("k_cache and v_cache shapes must match")
    if num_qo_heads != NUM_QO_HEADS:
        raise AssertionError("num_qo_heads must be 32")
    if num_kv_heads != NUM_KV_HEADS:
        raise AssertionError("num_kv_heads must be 4")
    if head_dim != HEAD_DIM or head_dim_k != HEAD_DIM:
        raise AssertionError("head_dim must be 128")
    if page_size != PAGE_SIZE:
        raise AssertionError("page_size must be 1")

    if kv_indptr.ndim != 1:
        raise ValueError("kv_indptr must be 1-D")
    if kv_indices.ndim != 1:
        raise ValueError("kv_indices must be 1-D")
    len_indptr = kv_indptr.shape[0]
    num_kv_indices = kv_indices.shape[0]
    if len_indptr != batch_size + 1:
        raise AssertionError("len_indptr must be batch_size + 1")
    # kv_indptr[-1] value (synchronize to host once)
    last_ind = int(kv_indptr[-1].detach().cpu().item())
    if num_kv_indices != last_ind:
        raise AssertionError("num_kv_indices must equal kv_indptr[-1].item()")

    # Dtypes
    if q.dtype != torch.bfloat16:
        raise TypeError("q must be bfloat16")
    if k_cache.dtype != torch.bfloat16 or v_cache.dtype != torch.bfloat16:
        raise TypeError("k_cache and v_cache must be bfloat16")
    if kv_indptr.dtype != torch.int32 or kv_indices.dtype != torch.int32:
        raise TypeError("kv_indptr and kv_indices must be int32")

    # Determine working device
    if not torch.cuda.is_available():
        for t in (q, k_cache, v_cache, kv_indptr, kv_indices):
            if t.is_cuda:
                raise RuntimeError("Input tensor is on CUDA device, but CUDA is not available.")
        raise RuntimeError("CUDA is not available. Triton kernel cannot run on CPU.")

    if q.device.type == "cuda":
        work_device = q.device
    elif k_cache.device.type == "cuda":
        work_device = k_cache.device
    elif v_cache.device.type == "cuda":
        work_device = v_cache.device
    elif kv_indptr.device.type == "cuda":
        work_device = kv_indptr.device
    elif kv_indices.device.type == "cuda":
        work_device = kv_indices.device
    else:
        work_device = torch.device("cuda")

    orig_q_device = q.device
    # Move tensors to working device
    q_dev = _ensure_cuda(q.contiguous(), work_device)
    k_cache_dev = _ensure_cuda(k_cache.contiguous(), work_device)
    v_cache_dev = _ensure_cuda(v_cache.contiguous(), work_device)
    kv_indptr_dev = _ensure_cuda(kv_indptr.contiguous(), work_device)
    kv_indices_dev = _ensure_cuda(kv_indices.contiguous(), work_device)

    # Allocate outputs on working device
    output_dev = torch.empty((batch_size, NUM_QO_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=work_device)
    lse_dev = torch.empty((batch_size, NUM_QO_HEADS), dtype=torch.float32, device=work_device)

    # Launch kernel
    BLOCK_M = 128  # tokens per block
    grid = (batch_size * NUM_QO_HEADS,)

    gqa_paged_decode_h32_kv4_d128_ps1_kernel[grid](
        q_dev,
        k_cache_dev,
        v_cache_dev,
        kv_indptr_dev,
        kv_indices_dev,
        sm_scale_val,
        output_dev,
        lse_dev,
        batch_size,
        NUM_QO_HEADS,
        # q strides
        q_dev.stride(0), q_dev.stride(1), q_dev.stride(2),
        # k strides
        k_cache_dev.stride(0), k_cache_dev.stride(1), k_cache_dev.stride(2), k_cache_dev.stride(3),
        # v strides
        v_cache_dev.stride(0), v_cache_dev.stride(1), v_cache_dev.stride(2), v_cache_dev.stride(3),
        # output strides
        output_dev.stride(0), output_dev.stride(1), output_dev.stride(2),
        # lse strides
        lse_dev.stride(0), lse_dev.stride(1),
        BLOCK_M=BLOCK_M,
        D_HEAD=HEAD_DIM,
        GQA_RATIO=GQA_RATIO,
        num_warps=8,
        num_stages=2,
    )

    # Move outputs back to original device of q
    if orig_q_device != work_device:
        output = output_dev.to(orig_q_device)
        lse = lse_dev.to(orig_q_device)
    else:
        output = output_dev
        lse = lse_dev

    return output, lse