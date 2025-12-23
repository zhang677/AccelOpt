# -------------------------------------------------------------
#  Optimized Triton kernel  (multi‑head per block)
#  – merges HEADS_PER_BLOCK heads inside a single block
#  – keeps the original precision (bf16 for I/O, fp32 inside)
# -------------------------------------------------------------
import math
import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------
#  Autotuned launch‑parameter configuration
# ------------------------------------------------------------------
HEADS_PER_BLOCK_DEFAULT = 4

@triton.autotune(
    configs=[
        # fast configs (2‑stage)
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_DC": 128, "BLOCK_DK": 64, "HEADS_PER_BLOCK": HEADS_PER_BLOCK_DEFAULT},
            num_warps=8,
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_N": 64, "BLOCK_DC": 128, "BLOCK_DK": 64, "HEADS_PER_BLOCK": HEADS_PER_BLOCK_DEFAULT},
            num_warps=4,
            num_stages=2,
        ),
        # larger KV‑tile with 3‑stage pipeline
        triton.Config(
            {"BLOCK_N": 256, "BLOCK_DC": 128, "BLOCK_DK": 64, "HEADS_PER_BLOCK": HEADS_PER_BLOCK_DEFAULT},
            num_warps=8,
            num_stages=3,
        ),
    ],
    key=["total_q", "H", "D_CKV", "D_KPE"],
)
@triton.jit
def mla_paged_prefill_causal_h16_ckv512_kpe64_ps1_kernel(
    q_nope_ptr,          # bf16   [total_q, H, D_CKV]
    q_pe_ptr,            # bf16   [total_q, H, D_KPE]
    ckv_ptr,             # bf16   [num_pages, D_CKV]
    kpe_ptr,             # bf16   [num_pages, D_KPE]
    qo_indptr_ptr,       # i32    [len_indptr]
    kv_indptr_ptr,       # i32    [len_indptr]
    kv_indices_ptr,      # i32    [num_kv_indices]
    q_to_seq_ptr,        # i32    [total_q]
    sm_scale,            # f32
    output_ptr,          # bf16   [total_q, H, D_CKV]
    lse_ptr,             # f32    [total_q, H]
    total_q,
    len_indptr,
    num_pages,
    num_kv_indices,
    H: tl.constexpr,           # =16
    D_CKV: tl.constexpr,       # =512
    D_KPE: tl.constexpr,       # =64
    BLOCK_N: tl.constexpr,     # KV‑tile size (64/128/256)
    BLOCK_DC: tl.constexpr,    # sub‑tile of CKV (4×128 = 512)
    BLOCK_DK: tl.constexpr,    # sub‑tile of KPE (covers whole KPE)
    HEADS_PER_BLOCK: tl.constexpr,
):
    # ------------------------------------------------------------------
    #  Program identifier (single‑dimensional)
    # ------------------------------------------------------------------
    pid = tl.program_id(0)                 # linear block id

    # Number of head‑blocks per query
    h_blocks = (H + HEADS_PER_BLOCK - 1) // HEADS_PER_BLOCK

    # --------------------------------------------------------------
    #  Decode which query and which head‑block this program works on
    # --------------------------------------------------------------
    qid          = pid // h_blocks                      # [0, total_q)
    head_block   = pid % h_blocks
    head_start   = head_block * HEADS_PER_BLOCK
    head_end     = tl.minimum(head_start + HEADS_PER_BLOCK, H)

    # Early‑exit if this block is out of range (guard against padding)
    if qid >= total_q:
        return

    # --------------------------------------------------------------
    #  Sequence (batch) id for this query
    # --------------------------------------------------------------
    seq_id = tl.load(q_to_seq_ptr + qid).to(tl.int32)

    # --------------------------------------------------------------
    #  Query range of the sequence
    # --------------------------------------------------------------
    q_start = tl.load(qo_indptr_ptr + seq_id).to(tl.int32)
    q_end   = tl.load(qo_indptr_ptr + seq_id + 1).to(tl.int32)
    q_len   = q_end - q_start
    q_rel   = qid - q_start                 # position of this query inside its seq

    # --------------------------------------------------------------
    #  KV range of the sequence
    # --------------------------------------------------------------
    kv_beg = tl.load(kv_indptr_ptr + seq_id).to(tl.int32)
    kv_end = tl.load(kv_indptr_ptr + seq_id + 1).to(tl.int32)
    kv_len = kv_end - kv_beg

    # Early‑exit flags (same for every head in the block)
    do_work = (kv_len > 0) & (q_len > 0)

    # --------------------------------------------------------------
    #  Causal‑mask parameters (same for every head)
    # --------------------------------------------------------------
    prefix_len    = kv_len - q_len                # tokens already cached
    query_abs_pos = prefix_len + q_rel            # absolute position of this query

    # ------------------------------------------------------------------
    #  Streaming softmax (base‑2) helpers
    # ------------------------------------------------------------------
    inv_ln2 = 1.4426950408889634          # 1 / ln(2)

    # ------------------------------------------------------------------
    #  Loop over the heads that belong to this block
    # ------------------------------------------------------------------
    for h in range(head_start, head_end):
        # --------------------------------------------------------------
        #  Load Q for this head (hoisted – reused for all KV tiles)
        # --------------------------------------------------------------
        qn_base = (qid * H + h) * D_CKV        # start of q_nope row
        qp_base = (qid * H + h) * D_KPE        # start of q_pe   row

        qn_chunk0 = tl.load(q_nope_ptr + qn_base + tl.arange(0, BLOCK_DC),
                            mask=True, other=0).to(tl.float32)
        qn_chunk1 = tl.load(q_nope_ptr + qn_base + BLOCK_DC + tl.arange(0, BLOCK_DC),
                            mask=True, other=0).to(tl.float32)
        qn_chunk2 = tl.load(q_nope_ptr + qn_base + 2 * BLOCK_DC + tl.arange(0, BLOCK_DC),
                            mask=True, other=0).to(tl.float32)
        qn_chunk3 = tl.load(q_nope_ptr + qn_base + 3 * BLOCK_DC + tl.arange(0, BLOCK_DC),
                            mask=True, other=0).to(tl.float32)

        qp_full = tl.load(q_pe_ptr + qp_base + tl.arange(0, D_KPE),
                         mask=True, other=0).to(tl.float32)

        # --------------------------------------------------------------
        #  Output / softmax accumulators for this head
        # --------------------------------------------------------------
        O0 = tl.zeros((BLOCK_DC,), dtype=tl.float32)
        O1 = tl.zeros((BLOCK_DC,), dtype=tl.float32)
        O2 = tl.zeros((BLOCK_DC,), dtype=tl.float32)
        O3 = tl.zeros((BLOCK_DC,), dtype=tl.float32)

        m_i2 = -float("inf")          # current max (base‑2)
        l_i2 = 0.0                     # sum of exp2 shifted by m_i2

        # --------------------------------------------------------------
        #  Main KV‑tile loop
        # --------------------------------------------------------------
        start = tl.zeros([], dtype=tl.int32)
        while start < kv_len:
            # ---- Load KV indices for the current tile ---------------------------------
            offs_n   = start + tl.arange(0, BLOCK_N)                # token offsets inside KV
            mask_n   = offs_n < kv_len                               # out‑of‑range guard
            page_idx = tl.load(kv_indices_ptr + kv_beg + offs_n,
                               mask=mask_n, other=0).to(tl.int32)   # token ids

            # ---- Load the four CKV sub‑tiles (once per tile) -------------------------
            ckv_tile0 = tl.load(
                ckv_ptr + page_idx[:, None] * D_CKV + tl.arange(0, BLOCK_DC)[None, :],
                mask=mask_n[:, None], other=0).to(tl.float32)

            ckv_tile1 = tl.load(
                ckv_ptr + page_idx[:, None] * D_CKV + (BLOCK_DC + tl.arange(0, BLOCK_DC))[None, :],
                mask=mask_n[:, None], other=0).to(tl.float32)

            ckv_tile2 = tl.load(
                ckv_ptr + page_idx[:, None] * D_CKV + (2 * BLOCK_DC + tl.arange(0, BLOCK_DC))[None, :],
                mask=mask_n[:, None], other=0).to(tl.float32)

            ckv_tile3 = tl.load(
                ckv_ptr + page_idx[:, None] * D_CKV + (3 * BLOCK_DC + tl.arange(0, BLOCK_DC))[None, :],
                mask=mask_n[:, None], other=0).to(tl.float32)

            # ---- Compute logits = q_nope·ckv + q_pe·kpe --------------------------------
            logits_tile = tl.zeros((BLOCK_N,), dtype=tl.float32)

            logits_tile += tl.sum(ckv_tile0 * qn_chunk0[None, :], axis=1)
            logits_tile += tl.sum(ckv_tile1 * qn_chunk1[None, :], axis=1)
            logits_tile += tl.sum(ckv_tile2 * qn_chunk2[None, :], axis=1)
            logits_tile += tl.sum(ckv_tile3 * qn_chunk3[None, :], axis=1)

            kp_tile = tl.load(
                kpe_ptr + page_idx[:, None] * D_KPE + tl.arange(0, D_KPE)[None, :],
                mask=mask_n[:, None], other=0).to(tl.float32)
            logits_tile += tl.sum(kp_tile * qp_full[None, :], axis=1)

            # ---- Scale, causal mask, streaming softmax (base‑2) -----------------------
            logits_tile = logits_tile * sm_scale
            z_tile = logits_tile * inv_ln2

            causal_mask = offs_n <= query_abs_pos
            valid_mask  = mask_n & causal_mask
            z_tile = tl.where(valid_mask, z_tile, -float("inf"))

            # max for the tile (base‑2)
            m_tile = tl.max(z_tile, axis=0)
            m_new  = tl.maximum(m_i2, m_tile)

            alpha = tl.where(m_i2 == -float("inf"), 0.0, tl.exp2(m_i2 - m_new))

            p_tile = tl.exp2(z_tile - m_new)
            p_tile = tl.where(valid_mask, p_tile, 0.0)

            sum_p = tl.sum(p_tile, axis=0)

            # update running max / sum
            l_i2 = l_i2 * alpha + sum_p
            O0 = O0 * alpha
            O1 = O1 * alpha
            O2 = O2 * alpha
            O3 = O3 * alpha
            m_i2 = m_new

            # ---- Accumulate weighted CKV values ------------------------------------
            O0 += tl.sum(p_tile[:, None] * ckv_tile0, axis=0)
            O1 += tl.sum(p_tile[:, None] * ckv_tile1, axis=0)
            O2 += tl.sum(p_tile[:, None] * ckv_tile2, axis=0)
            O3 += tl.sum(p_tile[:, None] * ckv_tile3, axis=0)

            # --------------------------------------------------------------
            #  Advance to the next KV tile
            # --------------------------------------------------------------
            start += BLOCK_N

        # ------------------------------------------------------------------
        #  Write results back (only if there is work for this head)
        # ------------------------------------------------------------------
        if do_work:
            # lse in base‑2 (m_i2 + log2(l_i2))
            lse_val = m_i2 + tl.log2(l_i2)

            inv_l = 1.0 / l_i2
            O0 = O0 * inv_l
            O1 = O1 * inv_l
            O2 = O2 * inv_l
            O3 = O3 * inv_l

            out_base = (qid * H + h) * D_CKV

            tl.store(output_ptr + out_base + tl.arange(0, BLOCK_DC),               O0.to(tl.bfloat16), mask=True)
            tl.store(output_ptr + out_base + BLOCK_DC + tl.arange(0, BLOCK_DC),   O1.to(tl.bfloat16), mask=True)
            tl.store(output_ptr + out_base + 2*BLOCK_DC + tl.arange(0, BLOCK_DC), O2.to(tl.bfloat16), mask=True)
            tl.store(output_ptr + out_base + 3*BLOCK_DC + tl.arange(0, BLOCK_DC), O3.to(tl.bfloat16), mask=True)

            tl.store(lse_ptr + (qid * H + h), lse_val)


# -------------------------------------------------------------
#  Host‑side helper & launch wrapper (multi‑head per block)
# -------------------------------------------------------------
def run(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv_cache: torch.Tensor,
    kpe_cache: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float | None = None,
    heads_per_block: int = HEADS_PER_BLOCK_DEFAULT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Execute the fused‑ckv kernel with **head‑fusion**: each CUDA block processes
    ``heads_per_block`` attention heads for the same query token.

    Parameters
    ----------
    q_nope     : torch.Tensor  [total_q, H, D_CKV]  (bf16)
    q_pe       : torch.Tensor  [total_q, H, D_KPE]  (bf16)
    ckv_cache  : torch.Tensor  [num_pages, 1, D_CKV] (bf16)
    kpe_cache  : torch.Tensor  [num_pages, 1, D_KPE] (bf16)
    qo_indptr  : torch.Tensor  [len_indptr] (int32)
    kv_indptr  : torch.Tensor  [len_indptr] (int32)
    kv_indices : torch.Tensor  [num_kv_indices] (int32)
    sm_scale   : float | None, optional
                 Softmax scaling factor. If ``None`` the default
                 ``1 / sqrt(D_CKV)`` (as in the original implementation) is used.
    heads_per_block : int, optional
                 Number of heads fused inside a single block. Must divide the
                 total head dimension (default = 4).

    Returns
    -------
    output : torch.Tensor   (bfloat16) shape [total_q, H, D_CKV]
    lse    : torch.Tensor   (float32) shape [total_q, H]
    """
    # ------------------------------------------------------------------
    #  Utility: move to CUDA if input is on CPU
    # ------------------------------------------------------------------
    def to_cuda(t):
        if not t.is_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA not available – all inputs must be on GPU or CUDA must be installed."
                )
            return t.cuda()
        return t

    # ------------------------------------------------------------------
    #  Type / shape checks (mirrors baseline)
    # ------------------------------------------------------------------
    assert q_nope.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert q_pe.dtype   in (torch.bfloat16, torch.float16, torch.float32)
    assert ckv_cache.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert kpe_cache.dtype in (torch.bfloat16, torch.float16, torch.float32)
    assert qo_indptr.dtype == torch.int32
    assert kv_indptr.dtype == torch.int32
    assert kv_indices.dtype == torch.int32

    # ------------------------------------------------------------------
    #  Preserve original device (so we can copy back if caller used CPU)
    # ------------------------------------------------------------------
    orig_device = q_nope.device

    # ------------------------------------------------------------------
    #  Move everything to GPU and enforce contiguous layout + bf16 for
    #  the tensors that the kernel expects.
    # ------------------------------------------------------------------
    q_nope    = to_cuda(q_nope).contiguous().to(torch.bfloat16)
    q_pe      = to_cuda(q_pe).contiguous().to(torch.bfloat16)
    ckv_cache = to_cuda(ckv_cache).contiguous().to(torch.bfloat16)
    kpe_cache = to_cuda(kpe_cache).contiguous().to(torch.bfloat16)
    qo_indptr = to_cuda(qo_indptr).contiguous()
    kv_indptr = to_cuda(kv_indptr).contiguous()
    kv_indices = to_cuda(kv_indices).contiguous()

    # ------------------------------------------------------------------
    #  Shapes / constants
    # ------------------------------------------------------------------
    total_q, H, D_CKV = q_nope.shape
    D_KPE = q_pe.shape[-1]
    page_size = ckv_cache.shape[1]
    len_indptr = qo_indptr.shape[0]
    batch_size = len_indptr - 1
    num_pages = ckv_cache.shape[0]
    num_kv_indices = kv_indices.shape[0]

    assert H == 16, "num_qo_heads must be 16"
    assert D_CKV == 512, "head_dim_ckv must be 512"
    assert D_KPE == 64, "head_dim_kpe must be 64"
    assert page_size == 1, "page_size must be 1"
    assert total_q == int(qo_indptr[-1].item()), "total_q must equal qo_indptr[-1]"
    assert num_kv_indices == int(kv_indptr[-1].item()), "num_kv_indices must equal kv_indptr[-1]"

    # ------------------------------------------------------------------
    #  Squeeze the page dimension (page_size == 1)
    # ------------------------------------------------------------------
    ckv_squeezed = ckv_cache.squeeze(1)      # [num_pages, 512]
    kpe_squeezed = kpe_cache.squeeze(1)      # [num_pages, 64]

    # ------------------------------------------------------------------
    #  Default softmax scale (same as original reference if not supplied)
    # ------------------------------------------------------------------
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(float(D_CKV))
    sm_scale = float(sm_scale)

    # ------------------------------------------------------------------
    #  Build a mapping from each query index → sequence id (batch element)
    #  This is tiny work (O(total_q)) and is done on the host CPU.
    # ------------------------------------------------------------------
    q_to_seq = torch.empty((total_q,), dtype=torch.int32, device=q_nope.device)
    qo_indptr_cpu = qo_indptr.cpu().to(torch.int64)
    for b in range(batch_size):
        start = int(qo_indptr_cpu[b].item())
        end   = int(qo_indptr_cpu[b + 1].item())
        if end > start:
            q_to_seq[start:end] = b

    # ------------------------------------------------------------------
    #  Allocate output buffers
    # ------------------------------------------------------------------
    output = torch.zeros(
        (total_q, H, D_CKV), dtype=torch.bfloat16, device=q_nope.device
    )
    lse = torch.full(
        (total_q, H), -float("inf"), dtype=torch.float32, device=q_nope.device
    )

    # ------------------------------------------------------------------
    #  Launch configuration (grid = total_q * ceil(H / heads_per_block))
    # ------------------------------------------------------------------
    h_blocks_per_query = (H + heads_per_block - 1) // heads_per_block
    grid = (total_q * h_blocks_per_query,)

    # ------------------------------------------------------------------
    #  Call the Triton kernel
    # ------------------------------------------------------------------
    mla_paged_prefill_causal_h16_ckv512_kpe64_ps1_kernel[grid](
        q_nope,
        q_pe,
        ckv_squeezed,
        kpe_squeezed,
        qo_indptr,
        kv_indptr,
        kv_indices,
        q_to_seq,
        sm_scale,
        output,
        lse,
        total_q,
        len_indptr,
        num_pages,
        num_kv_indices,
        H=H,
        D_CKV=D_CKV,
        D_KPE=D_KPE,
        # BLOCK_* and HEADS_PER_BLOCK are injected automatically by @triton.autotune
    )

    # ------------------------------------------------------------------
    #  Copy back to original device if caller gave us CPU tensors
    # ------------------------------------------------------------------
    if orig_device.type != "cuda":
        output = output.to(orig_device)
        lse    = lse.to(orig_device)

    return output, lse