import torch
import triton
import triton.language as tl
import math
import inspect

# -------------------
# Triton Kernel
# -------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 256}, num_warps=4, num_stages=3),
    ],
    key=['total_q'],
)
@triton.jit
def _gqa_paged_prefill_causal_kernel(
    # Pointers to Tensors
    Q, K_cache, V_cache,
    QO_indptr, KV_indptr, KV_indices,
    Q_seq_idx_map,
    sm_scale,
    Output, LSE,
    # Stride Args
    stride_q_total, stride_q_head, stride_q_dim,
    stride_k_page, stride_k_ps, stride_k_head, stride_k_dim,
    stride_v_page, stride_v_ps, stride_v_head, stride_v_dim,
    # Metadata
    total_q,
    num_qo_heads,
    num_kv_heads,
    # Constexpr
    GQA_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    """
    Triton kernel for GQA paged prefill with causal masking.
    Each program computes the attention output for one query token and one query head.
    Grid: (total_q, num_qo_heads)
    """
    # 1. Get Program IDs for the current query token and head
    global_q_idx = tl.program_id(0)
    qo_head_idx = tl.program_id(1)

    # 2. Find sequence boundaries for the current query token using the precomputed map
    seq_idx = tl.load(Q_seq_idx_map + global_q_idx)
    q_start = tl.load(QO_indptr + seq_idx)
    q_end = tl.load(QO_indptr + seq_idx + 1)
    kv_start = tl.load(KV_indptr + seq_idx)
    kv_end = tl.load(KV_indptr + seq_idx + 1)

    # 3. Determine causal boundary for attention
    num_q_tokens = q_end - q_start
    num_kv_tokens = kv_end - kv_start
    q_idx_local = global_q_idx - q_start
    delta = num_kv_tokens - num_q_tokens
    max_kv_idx_for_q = q_idx_local + delta + 1

    # 4. Handle edge case where a query has no keys to attend to.
    if max_kv_idx_for_q <= 0:
        # Store default values (0 for output, -inf for LSE) and exit.
        output_ptr = Output + global_q_idx * stride_q_total + qo_head_idx * stride_q_head
        offs_d = tl.arange(0, HEAD_DIM)
        tl.store(output_ptr + offs_d, tl.zeros([HEAD_DIM], dtype=tl.bfloat16))

        lse_ptr = LSE + global_q_idx * num_qo_heads + qo_head_idx
        tl.store(lse_ptr, -float("inf"))
        return

    # 5. Load Q vector for the current query token and head
    offs_d = tl.arange(0, HEAD_DIM)
    q_offset = global_q_idx * stride_q_total + qo_head_idx * stride_q_head
    q_ptr = Q + q_offset
    q = tl.load(q_ptr + offs_d).to(tl.float32)

    # 6. Initialize accumulator, max logit, and lse for online softmax
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    m_i = -float("inf")
    l_i = 0.0

    # 7. Loop over KV cache blocks
    kv_head_idx = qo_head_idx // GQA_RATIO
    for n_offset in range(0, max_kv_idx_for_q, BLOCK_N):
        # a. Create indices and masks for the current KV block
        offs_n = n_offset + tl.arange(0, BLOCK_N)
        kv_indices_offs = kv_start + offs_n
        kv_mask = (offs_n < max_kv_idx_for_q)
        page_indices_mask = kv_mask & (kv_indices_offs < kv_end)

        # b. Load page indices from KV_indices global memory
        page_ids = tl.load(KV_indices + kv_indices_offs, mask=page_indices_mask, other=0)

        # c. Load K block using gathered page_ids
        k_ptr = K_cache + (page_ids[:, None] * stride_k_page +
                           kv_head_idx * stride_k_head +
                           offs_d[None, :])
        k = tl.load(k_ptr, mask=page_indices_mask[:, None], other=0.0).to(tl.float32)

        # d. FIX: Compute Q @ K^T scores using element-wise multiplication and reduction.
        # tl.dot is not suitable for matrix-vector products due to tensor core dimension constraints.
        s = tl.sum(k * q[None, :], axis=1)

        s *= sm_scale
        s = tl.where(kv_mask, s, -float("inf"))

        # e. Online softmax update
        m_ij = tl.maximum(m_i, tl.max(s, 0))
        p = tl.exp(s - m_ij)
        l_ij = tl.exp(m_i - m_ij) * l_i + tl.sum(p, 0)

        # f. Update accumulator (rescale previous accumulator)
        acc_scale = tl.exp(m_i - m_ij)
        acc = acc * acc_scale

        # Load V block, transposing it on the fly for calculation
        v_ptr_T = V_cache + (page_ids[None, :] * stride_v_page +
                             kv_head_idx * stride_v_head +
                             offs_d[:, None])
        v_T = tl.load(v_ptr_T, mask=page_indices_mask[None, :], other=0.0)

        p_typed = p.to(v_T.dtype)

        # FIX: Update accumulator using element-wise multiplication and reduction.
        # This correctly computes the weighted sum of value vectors (V.T @ p)
        acc_update = tl.sum(v_T * p_typed[None, :], axis=1)
        acc += acc_update

        # g. Update softmax stats for next iteration
        m_i = m_ij
        l_i = l_ij

    # 8. Finalize output and LSE
    o = acc / l_i
    lse_val = m_i + tl.log(l_i)
    lse_val *= 1.4426950408889634  # 1.0 / math.log(2)

    # 9. Store results to global memory
    output_ptr = Output + global_q_idx * stride_q_total + qo_head_idx * stride_q_head
    tl.store(output_ptr + offs_d, o.to(tl.bfloat16))

    lse_ptr = LSE + global_q_idx * num_qo_heads + qo_head_idx
    tl.store(lse_ptr, lse_val)


# -------------------
# Host-side Wrapper
# -------------------

def gqa_paged_prefill_causal_h32_kv8_d128_ps1(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float = None,
):
    """
    Computes Grouped-Query Attention for paged prefill phase with causal masking.

    Args:
        q (torch.Tensor): Query tensor of shape [total_q, num_qo_heads, head_dim].
        k_cache (torch.Tensor): Key cache tensor of shape [num_pages, page_size, num_kv_heads, head_dim].
        v_cache (torch.Tensor): Value cache tensor of shape [num_pages, page_size, num_kv_heads, head_dim].
        qo_indptr (torch.Tensor): Query offsets for each sequence of shape [batch_size + 1].
        kv_indptr (torch.Tensor): KV page offsets for each sequence of shape [batch_size + 1].
        kv_indices (torch.Tensor): Page IDs for KV cache lookups of shape [num_kv_indices].
        sm_scale (float, optional): Softmax scale. Defaults to 1/sqrt(head_dim).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - The attention output tensor of shape [total_q, num_qo_heads, head_dim].
            - The log-sum-exp of attention logits (base 2) of shape [total_q, num_qo_heads].
    """
    # 1. Validate inputs and extract dimensions
    assert q.dim() == 3, "q must be a 3D tensor"
    assert k_cache.dim() == 4, "k_cache must be a 4D tensor"
    assert v_cache.dim() == 4, "v_cache must be a 4D tensor"
    assert q.dtype == torch.bfloat16
    assert k_cache.dtype == torch.bfloat16
    assert v_cache.dtype == torch.bfloat16
    assert qo_indptr.dtype == torch.int32
    assert kv_indptr.dtype == torch.int32
    assert kv_indices.dtype == torch.int32

    total_q, num_qo_heads, head_dim = q.shape
    num_pages, page_size, num_kv_heads, _ = k_cache.shape

    # Constants from spec
    assert num_qo_heads == 32
    assert num_kv_heads == 8
    assert head_dim == 128
    assert page_size == 1

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # 2. Prepare outputs
    output = torch.empty_like(q)
    lse = torch.empty((total_q, num_qo_heads), device=q.device, dtype=torch.float32)

    # 3. Pre-compute a map from global query index to sequence index.
    # This avoids a slow and divergent search loop inside the kernel.
    batch_size = qo_indptr.numel() - 1
    q_seq_len = qo_indptr[1:] - qo_indptr[:-1]
    q_seq_idx_map = torch.arange(batch_size, device=q.device, dtype=torch.int32).repeat_interleave(q_seq_len)

    # 4. Set up grid and call kernel
    grid = (total_q, num_qo_heads)

    _gqa_paged_prefill_causal_kernel[grid](
        # Tensors
        q, k_cache, v_cache,
        qo_indptr, kv_indptr, kv_indices,
        q_seq_idx_map,
        sm_scale,
        output, lse,
        # Strides
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        # Metadata
        total_q,
        num_qo_heads,
        num_kv_heads,
        # Constexpr
        GQA_RATIO=num_qo_heads // num_kv_heads,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
    )

    return output, lse


# -------------------
# Entry Point
# -------------------

def run(*args, **kwargs):
    """
    Public entry point for the Triton kernel.
    Handles device management and calls the main logic.
    """
    # 1. Get the signature of the core logic function
    sig = inspect.signature(gqa_paged_prefill_causal_h32_kv8_d128_ps1)

    # 2. Bind the passed arguments to the signature
    try:
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
    except TypeError as e:
        raise TypeError(f"Error binding arguments: {e}") from e

    # 3. Extract tensor arguments and sm_scale
    all_args = bound_args.arguments
    q = all_args['q']
    k_cache = all_args['k_cache']
    v_cache = all_args['v_cache']
    qo_indptr = all_args['qo_indptr']
    kv_indptr = all_args['kv_indptr']
    kv_indices = all_args['kv_indices']
    sm_scale = all_args['sm_scale']

    # 4. Device management
    if not torch.cuda.is_available():
        raise RuntimeError("Triton kernel requires CUDA.")

    tensor_args = [q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices]

    # Determine the target device from the first available CUDA tensor, or default to 'cuda'
    target_device = 'cuda'
    for t in tensor_args:
        if isinstance(t, torch.Tensor) and t.is_cuda:
            target_device = t.device
            break

    # Store original devices to move results back later
    original_devices = [t.device for t in tensor_args]

    # Move all tensors to the target device
    moved_tensors = [t.to(target_device) for t in tensor_args]
    q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices = moved_tensors

    # 5. Call the kernel
    output, lse = gqa_paged_prefill_causal_h32_kv8_d128_ps1(
        q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale
    )

    # 6. Move results back to the original device of the 'q' tensor
    q_orig_device = original_devices[0]
    output = output.to(q_orig_device)
    lse = lse.to(q_orig_device)

    return output, lse