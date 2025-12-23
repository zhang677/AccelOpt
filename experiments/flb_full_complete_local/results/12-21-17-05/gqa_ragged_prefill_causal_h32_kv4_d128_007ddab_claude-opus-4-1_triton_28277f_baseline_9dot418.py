import torch
import triton
import triton.language as tl
import math

@triton.jit
def gqa_ragged_prefill_causal_kernel(
    q_ptr, k_ptr, v_ptr,
    output_ptr, lse_ptr,
    qo_indptr_ptr, kv_indptr_ptr,
    sm_scale,
    batch_idx,
    total_q, total_kv,
    stride_q_tok, stride_q_h, stride_q_d,
    stride_kv_tok, stride_kv_h, stride_kv_d,
    stride_out_tok, stride_out_h, stride_out_d,
    stride_lse_tok, stride_lse_h,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_QO_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_RATIO: tl.constexpr,
):
    # Get sequence boundaries
    q_start = tl.load(qo_indptr_ptr + batch_idx)
    q_end = tl.load(qo_indptr_ptr + batch_idx + 1)
    kv_start = tl.load(kv_indptr_ptr + batch_idx)
    kv_end = tl.load(kv_indptr_ptr + batch_idx + 1)
    
    num_q_tokens = q_end - q_start
    num_kv_tokens = kv_end - kv_start
    
    # Block indices
    block_m = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # Calculate KV head for GQA
    kv_head = head_idx // GQA_RATIO
    
    # Calculate query token indices for this block
    q_block_start = block_m * BLOCK_M
    
    # Early exit if out of bounds
    if q_block_start >= num_q_tokens:
        return
    
    # Calculate causal mask boundary
    delta = num_kv_tokens - num_q_tokens
    
    # Initialize accumulators for each query in the block
    m_i = tl.full([BLOCK_M], value=-float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # Load query block
    q_offs_m = tl.arange(0, BLOCK_M)
    q_offs_d = tl.arange(0, BLOCK_D)
    q_mask = (q_block_start + q_offs_m[:, None] < num_q_tokens) & (q_offs_d[None, :] < BLOCK_D)
    q_ptrs = q_ptr + (q_start + q_block_start + q_offs_m[:, None]) * stride_q_tok + head_idx * stride_q_h + q_offs_d[None, :] * stride_q_d
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)
    
    # Process KV blocks
    for kv_block_start in range(0, num_kv_tokens, BLOCK_N):
        # Load K block
        k_offs_n = tl.arange(0, BLOCK_N)
        k_offs_d = tl.arange(0, BLOCK_D)
        k_mask = (kv_block_start + k_offs_n[:, None] < num_kv_tokens) & (k_offs_d[None, :] < BLOCK_D)
        k_ptrs = k_ptr + (kv_start + kv_block_start + k_offs_n[:, None]) * stride_kv_tok + kv_head * stride_kv_h + k_offs_d[None, :] * stride_kv_d
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)
        
        # Compute QK^T
        qk = tl.dot(q, tl.trans(k))
        
        # Apply scale
        qk = qk * sm_scale
        
        # Apply causal mask
        q_offs_m_2 = tl.arange(0, BLOCK_M)
        k_offs_n_2 = tl.arange(0, BLOCK_N)
        
        # Calculate the maximum KV index each query can attend to
        q_positions = q_block_start + q_offs_m_2
        max_kv_idx = q_positions + 1 + delta
        
        # Create causal mask
        kv_positions = kv_block_start + k_offs_n_2
        causal_mask = kv_positions[None, :] < max_kv_idx[:, None]
        
        # Also ensure we don't go beyond actual tokens
        valid_q = (q_block_start + q_offs_m_2[:, None]) < num_q_tokens
        valid_kv = (kv_block_start + k_offs_n_2[None, :]) < num_kv_tokens
        qk_mask = causal_mask & valid_q & valid_kv
        
        qk = tl.where(qk_mask, qk, -float('inf'))
        
        # Online softmax update
        m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
        
        # Compute attention weights with numerical stability
        p = tl.exp(qk - m_i_new[:, None])
        
        # Update running sum with correction
        alpha = tl.exp(m_i - m_i_new)
        l_i_new = alpha * l_i + tl.sum(p, axis=1)
        
        # Load V block
        v_offs_n = tl.arange(0, BLOCK_N)
        v_offs_d = tl.arange(0, BLOCK_D)
        v_mask = (kv_block_start + v_offs_n[:, None] < num_kv_tokens) & (v_offs_d[None, :] < BLOCK_D)
        v_ptrs = v_ptr + (kv_start + kv_block_start + v_offs_n[:, None]) * stride_kv_tok + kv_head * stride_kv_h + v_offs_d[None, :] * stride_kv_d
        v = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)
        
        # Update accumulator with proper scaling
        acc = acc * alpha[:, None]
        acc += tl.dot(p, v)
        
        # Update running max and sum
        m_i = m_i_new
        l_i = l_i_new
    
    # Normalize output
    acc = acc / l_i[:, None]
    
    # Store output
    out_offs_m = tl.arange(0, BLOCK_M)
    out_offs_d = tl.arange(0, BLOCK_D)
    out_mask = (q_block_start + out_offs_m[:, None] < num_q_tokens) & (out_offs_d[None, :] < BLOCK_D)
    out_ptrs = output_ptr + (q_start + q_block_start + out_offs_m[:, None]) * stride_out_tok + head_idx * stride_out_h + out_offs_d[None, :] * stride_out_d
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)
    
    # Store LSE (log2 scale)
    lse_offs = tl.arange(0, BLOCK_M)
    lse_mask = q_block_start + lse_offs < num_q_tokens
    lse_ptrs = lse_ptr + (q_start + q_block_start + lse_offs) * stride_lse_tok + head_idx * stride_lse_h
    # Convert to log2 scale
    log2_e = 1.4426950408889634  # 1.0 / ln(2)
    lse_val = (m_i + tl.log(l_i)) * log2_e
    tl.store(lse_ptrs, lse_val, mask=lse_mask)

def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    # Handle device management
    device = None
    inputs_on_cuda = []
    original_devices = []
    
    # Check and move tensors to CUDA if needed
    for tensor, name in [(q, 'q'), (k, 'k'), (v, 'v'), (qo_indptr, 'qo_indptr'), (kv_indptr, 'kv_indptr')]:
        original_devices.append(tensor.device)
        if tensor.is_cuda:
            if device is None:
                device = tensor.device
            inputs_on_cuda.append(True)
        else:
            if not torch.cuda.is_available():
                raise RuntimeError(f"Tensor '{name}' is on CPU but CUDA is not available")
            if device is None:
                device = torch.device('cuda')
            inputs_on_cuda.append(False)
    
    # Move CPU tensors to GPU
    if not q.is_cuda:
        q = q.cuda()
    if not k.is_cuda:
        k = k.cuda()
    if not v.is_cuda:
        v = v.cuda()
    if not qo_indptr.is_cuda:
        qo_indptr = qo_indptr.cuda()
    if not kv_indptr.is_cuda:
        kv_indptr = kv_indptr.cuda()
    
    # Get dimensions
    total_q, num_qo_heads, head_dim = q.shape
    total_kv, num_kv_heads, _ = k.shape
    len_indptr = qo_indptr.shape[0]
    
    # Verify constants
    assert num_qo_heads == 32
    assert num_kv_heads == 4
    assert head_dim == 128
    
    # Verify constraints
    assert total_q == qo_indptr[-1].item()
    assert total_kv == kv_indptr[-1].item()
    
    # Initialize outputs
    output = torch.zeros((total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device)
    lse = torch.full((total_q, num_qo_heads), -float('inf'), dtype=torch.float32, device=device)
    
    # Constants optimized for B200
    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_D = 128
    GQA_RATIO = num_qo_heads // num_kv_heads
    
    # Process each batch
    for batch_idx in range(len_indptr - 1):
        q_start = qo_indptr[batch_idx].item()
        q_end = qo_indptr[batch_idx + 1].item()
        kv_start = kv_indptr[batch_idx].item()
        kv_end = kv_indptr[batch_idx + 1].item()
        
        num_q_tokens = q_end - q_start
        num_kv_tokens = kv_end - kv_start
        
        if num_q_tokens <= 0 or num_kv_tokens <= 0:
            continue
        
        grid = (triton.cdiv(num_q_tokens, BLOCK_M), num_qo_heads)
        
        gqa_ragged_prefill_causal_kernel[grid](
            q, k, v,
            output, lse,
            qo_indptr, kv_indptr,
            sm_scale,
            batch_idx,
            total_q, total_kv,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            lse.stride(0), lse.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
            NUM_QO_HEADS=num_qo_heads,
            NUM_KV_HEADS=num_kv_heads,
            GQA_RATIO=GQA_RATIO,
            num_warps=4,
            num_stages=2,
        )
    
    # Move outputs back to original device if needed
    if not inputs_on_cuda[0]:  # q was originally on CPU
        output = output.to(original_devices[0])
        lse = lse.to(original_devices[0])
    
    return output, lse