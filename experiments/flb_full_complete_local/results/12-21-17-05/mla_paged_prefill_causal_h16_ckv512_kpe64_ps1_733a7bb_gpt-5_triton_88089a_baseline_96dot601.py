import math
import torch
import triton
import triton.language as tl


@triton.jit
def mla_paged_prefill_causal_h16_ckv512_kpe64_ps1_kernel(
    q_nope_ptr,  # bf16 [total_q, H, D_CKV]
    q_pe_ptr,    # bf16 [total_q, H, D_KPE]
    ckv_ptr,     # bf16 [num_pages, D_CKV]
    kpe_ptr,     # bf16 [num_pages, D_KPE]
    qo_indptr_ptr,  # int32 [len_indptr]
    kv_indptr_ptr,  # int32 [len_indptr]
    kv_indices_ptr,  # int32 [num_kv_indices]
    q_to_seq_ptr,   # int32 [total_q]
    sm_scale,       # float32
    output_ptr,     # bf16 [total_q, H, D_CKV]
    lse_ptr,        # float32 [total_q, H]
    total_q,
    len_indptr,
    num_pages,
    num_kv_indices,
    H: tl.constexpr,
    D_CKV: tl.constexpr,
    D_KPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DC: tl.constexpr,
    BLOCK_DK: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_h >= H:
        return
    if pid_q >= total_q:
        return

    # Load sequence id for this query
    seq_id = tl.load(q_to_seq_ptr + pid_q, mask=True, other=0).to(tl.int32)

    # Load q range for this seq
    q_start = tl.load(qo_indptr_ptr + seq_id, mask=True, other=0).to(tl.int32)
    q_end = tl.load(qo_indptr_ptr + seq_id + 1, mask=True, other=0).to(tl.int32)
    q_len = q_end - q_start
    q_rel = pid_q - q_start

    # Load kv range for this seq
    kv_beg = tl.load(kv_indptr_ptr + seq_id, mask=True, other=0).to(tl.int32)
    kv_end = tl.load(kv_indptr_ptr + seq_id + 1, mask=True, other=0).to(tl.int32)
    kv_len = kv_end - kv_beg

    # Early exit flags
    do_work = (kv_len > 0) & (q_len > 0)

    # Causal mask parameters
    prefix_len = kv_len - q_len
    query_abs_pos = prefix_len + q_rel

    # Base pointers for Q (row-major [Q, H, D])
    qn_base = (pid_q * H + pid_h) * D_CKV
    qp_base = (pid_q * H + pid_h) * D_KPE

    # Streaming softmax in base-2
    inv_ln2 = 1.4426950408889634  # 1 / ln(2)
    m_i2 = -float("inf")
    l_i2 = 0.0

    # Accumulators for output: split into 4 chunks of 128 dims each
    O0 = tl.zeros((BLOCK_DC,), dtype=tl.float32)
    O1 = tl.zeros((BLOCK_DC,), dtype=tl.float32)
    O2 = tl.zeros((BLOCK_DC,), dtype=tl.float32)
    O3 = tl.zeros((BLOCK_DC,), dtype=tl.float32)

    start = tl.zeros((), dtype=tl.int32)
    while start < kv_len:
        # KV indices for this tile
        offs_n = start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < kv_len
        page_idx = tl.load(kv_indices_ptr + kv_beg + offs_n, mask=mask_n, other=0).to(tl.int32)

        # Compute logits for this tile
        logits_tile = tl.zeros((BLOCK_N,), dtype=tl.float32)

        # q_nope dot ckv
        for d0 in tl.static_range(0, D_CKV, BLOCK_DC):
            d_offsets = d0 + tl.arange(0, BLOCK_DC)
            qn_chunk = tl.load(q_nope_ptr + qn_base + d_offsets, mask=True, other=0).to(tl.float32)
            kc_tile = tl.load(
                ckv_ptr + page_idx[:, None] * D_CKV + d_offsets[None, :],
                mask=mask_n[:, None],
                other=0
            ).to(tl.float32)
            logits_tile += tl.sum(kc_tile * qn_chunk[None, :], axis=1)

        # q_pe dot kpe
        for d0 in tl.static_range(0, D_KPE, BLOCK_DK):
            d_offsets = d0 + tl.arange(0, BLOCK_DK)
            qp_chunk = tl.load(q_pe_ptr + qp_base + d_offsets, mask=True, other=0).to(tl.float32)
            kp_tile = tl.load(
                kpe_ptr + page_idx[:, None] * D_KPE + d_offsets[None, :],
                mask=mask_n[:, None],
                other=0
            ).to(tl.float32)
            logits_tile += tl.sum(kp_tile * qp_chunk[None, :], axis=1)

        # Scale logits
        logits_tile = logits_tile * sm_scale
        # Convert to base-2 space
        z_tile = logits_tile * inv_ln2

        # Causal mask: allow only indices <= query_abs_pos
        causal_mask = offs_n <= query_abs_pos
        valid_mask = mask_n & causal_mask
        neg_inf = -float("inf")
        z_tile = tl.where(valid_mask, z_tile, neg_inf)

        # Streaming softmax update (base-2)
        m_tile = tl.max(z_tile, axis=0)
        m_new = tl.maximum(m_i2, m_tile)
        # Guard against -inf - -inf
        alpha = tl.where(m_i2 == -float("inf"), 0.0, tl.exp2(m_i2 - m_new))
        p_tile = tl.exp2(z_tile - m_new)
        p_tile = tl.where(valid_mask, p_tile, 0.0)
        sum_p = tl.sum(p_tile, axis=0)

        # Update l and O with scaling
        l_i2 = l_i2 * alpha + sum_p
        O0 = O0 * alpha
        O1 = O1 * alpha
        O2 = O2 * alpha
        O3 = O3 * alpha

        # Accumulate O chunks: O += sum_n p_tile[n] * Kc[n, :]
        # Chunk 0
        d_offsets0 = 0 + tl.arange(0, BLOCK_DC)
        kc0 = tl.load(
            ckv_ptr + page_idx[:, None] * D_CKV + d_offsets0[None, :],
            mask=mask_n[:, None],
            other=0
        ).to(tl.float32)
        O0 += tl.sum(p_tile[:, None] * kc0, axis=0)

        # Chunk 1
        d_offsets1 = BLOCK_DC + tl.arange(0, BLOCK_DC)
        kc1 = tl.load(
            ckv_ptr + page_idx[:, None] * D_CKV + d_offsets1[None, :],
            mask=mask_n[:, None],
            other=0
        ).to(tl.float32)
        O1 += tl.sum(p_tile[:, None] * kc1, axis=0)

        # Chunk 2
        d_offsets2 = (2 * BLOCK_DC) + tl.arange(0, BLOCK_DC)
        kc2 = tl.load(
            ckv_ptr + page_idx[:, None] * D_CKV + d_offsets2[None, :],
            mask=mask_n[:, None],
            other=0
        ).to(tl.float32)
        O2 += tl.sum(p_tile[:, None] * kc2, axis=0)

        # Chunk 3
        d_offsets3 = (3 * BLOCK_DC) + tl.arange(0, BLOCK_DC)
        kc3 = tl.load(
            ckv_ptr + page_idx[:, None] * D_CKV + d_offsets3[None, :],
            mask=mask_n[:, None],
            other=0
        ).to(tl.float32)
        O3 += tl.sum(p_tile[:, None] * kc3, axis=0)

        m_i2 = m_new
        start += BLOCK_N

    # Finalize and store
    if do_work:
        # lse in base-2
        lse_val = m_i2 + tl.log2(l_i2)
        # Normalize output
        inv_l = 1.0 / l_i2
        O0 = O0 * inv_l
        O1 = O1 * inv_l
        O2 = O2 * inv_l
        O3 = O3 * inv_l

        # Store O to output bf16
        out_base = (pid_q * H + pid_h) * D_CKV
        # Chunk 0
        d_offsets0 = 0 + tl.arange(0, BLOCK_DC)
        tl.store(output_ptr + out_base + d_offsets0, O0.to(tl.bfloat16), mask=True)
        # Chunk 1
        d_offsets1 = BLOCK_DC + tl.arange(0, BLOCK_DC)
        tl.store(output_ptr + out_base + d_offsets1, O1.to(tl.bfloat16), mask=True)
        # Chunk 2
        d_offsets2 = (2 * BLOCK_DC) + tl.arange(0, BLOCK_DC)
        tl.store(output_ptr + out_base + d_offsets2, O2.to(tl.bfloat16), mask=True)
        # Chunk 3
        d_offsets3 = (3 * BLOCK_DC) + tl.arange(0, BLOCK_DC)
        tl.store(output_ptr + out_base + d_offsets3, O3.to(tl.bfloat16), mask=True)

        # Store lse
        lse_off = pid_q * H + pid_h
        tl.store(lse_ptr + lse_off, lse_val)


def run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale=None):
    # Validate CUDA availability and manage devices
    def to_cuda_if_needed(t):
        if not t.is_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available, but input tensors are on CPU. Please enable CUDA or move inputs to GPU.")
            return t.cuda()
        return t

    # Dtypes and shapes
    assert q_nope.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert q_pe.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert ckv_cache.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert kpe_cache.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert qo_indptr.dtype == torch.int32
    assert kv_indptr.dtype == torch.int32
    assert kv_indices.dtype == torch.int32

    # Original device for outputs
    orig_device = q_nope.device

    # Move to CUDA if needed
    q_nope = to_cuda_if_needed(q_nope)
    q_pe = to_cuda_if_needed(q_pe)
    ckv_cache = to_cuda_if_needed(ckv_cache)
    kpe_cache = to_cuda_if_needed(kpe_cache)
    qo_indptr = to_cuda_if_needed(qo_indptr)
    kv_indptr = to_cuda_if_needed(kv_indptr)
    kv_indices = to_cuda_if_needed(kv_indices)

    # Ensure contiguous layouts and correct dtypes (bf16 for caches/Q)
    q_nope = q_nope.to(torch.bfloat16).contiguous()
    q_pe = q_pe.to(torch.bfloat16).contiguous()
    ckv_cache = ckv_cache.to(torch.bfloat16).contiguous()
    kpe_cache = kpe_cache.to(torch.bfloat16).contiguous()
    qo_indptr = qo_indptr.contiguous()
    kv_indptr = kv_indptr.contiguous()
    kv_indices = kv_indices.contiguous()

    # Shapes and constants
    total_q, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    page_size = ckv_cache.shape[1]
    len_indptr = qo_indptr.shape[0]
    batch_size = len_indptr - 1
    num_pages = ckv_cache.shape[0]
    num_kv_indices = kv_indices.shape[0]

    # Checks for constants
    assert num_qo_heads == 16, "num_qo_heads must be 16"
    assert head_dim_ckv == 512, "head_dim_ckv must be 512"
    assert head_dim_kpe == 64, "head_dim_kpe must be 64"
    assert page_size == 1, "page_size must be 1"
    assert total_q == int(qo_indptr[-1].item()), "total_q must equal qo_indptr[-1]"
    assert num_kv_indices == int(kv_indptr[-1].item()), "num_kv_indices must equal kv_indptr[-1]"

    # Squeeze page dimension as page_size == 1
    ckv_squeezed = ckv_cache.squeeze(1).contiguous()  # [num_pages, 512]
    kpe_squeezed = kpe_cache.squeeze(1).contiguous()  # [num_pages, 64]

    # Default sm_scale
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(float(head_dim_ckv))
    sm_scale = float(sm_scale)

    # Build mapping from query index to sequence id
    q_to_seq = torch.empty((total_q,), dtype=torch.int32, device=q_nope.device)
    qo_indptr_cpu = qo_indptr.to(torch.int64).cpu()
    for b in range(batch_size):
        start = int(qo_indptr_cpu[b].item())
        end = int(qo_indptr_cpu[b + 1].item())
        if end > start:
            q_to_seq[start:end] = b

    # Allocate outputs on device, initialize as in reference
    output = torch.zeros((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=q_nope.device)
    lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=q_nope.device)

    # Launch kernel
    H = 16
    D_CKV = 512
    D_KPE = 64
    BLOCK_N = 128
    BLOCK_DC = 128
    BLOCK_DK = 64

    grid = (total_q, H)

    mla_paged_prefill_causal_h16_ckv512_kpe64_ps1_kernel[grid](
        q_nope,  # bf16
        q_pe,    # bf16
        ckv_squeezed,  # bf16
        kpe_squeezed,  # bf16
        qo_indptr,     # int32
        kv_indptr,     # int32
        kv_indices,    # int32
        q_to_seq,      # int32
        sm_scale,      # float32
        output,        # bf16
        lse,           # float32
        total_q,
        len_indptr,
        num_pages,
        num_kv_indices,
        H=H,
        D_CKV=D_CKV,
        D_KPE=D_KPE,
        BLOCK_N=BLOCK_N,
        BLOCK_DC=BLOCK_DC,
        BLOCK_DK=BLOCK_DK,
        num_warps=8,
        num_stages=2,
    )

    # Move results back to the original device if necessary
    if orig_device.type != 'cuda':
        output = output.to(orig_device)
        lse = lse.to(orig_device)

    return output, lse