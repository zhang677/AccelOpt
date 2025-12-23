# -------------------------------------------------------------
#  Triton implementation of ragged GQA causal pre‑fill attention
#  (BF16 activations, FP32 accumulation, additive‑bias mask)
# -------------------------------------------------------------
import torch
import triton
import triton.language as tl
import math


# -------------------------------------------------------------
#  1️⃣  Autotuned kernel
# -------------------------------------------------------------
@triton.autotune(
    configs=[
        # ---- Existing safe configs ---------------------------------
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 32, "num_warps": 2, "num_stages": 2},
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2},
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 8, "num_stages": 2},
            num_stages=2,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "num_warps": 8, "num_stages": 2},
            num_stages=2,
        ),
        # ---- Deeper‑pipeline configs ---------------------------------
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "num_warps": 16, "num_stages": 3},
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "num_warps": 16, "num_stages": 3},
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "num_warps": 16, "num_stages": 3},
            num_stages=3,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "num_warps": 16, "num_stages": 3},
            num_stages=3,
        ),
        # ---- NEW aggressive configs ---------------------------------
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "num_warps": 32, "num_stages": 4},
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "num_warps": 32, "num_stages": 4},
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 512, "num_warps": 32, "num_stages": 4},
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 512, "num_warps": 32, "num_stages": 4},
            num_stages=4,
        ),
    ],
    key=["total_q", "total_kv", "num_qo_heads", "num_kv_heads"],
)
@triton.jit
def gqa_ragged_prefill_causal_kernel(
    # -------------------------------------------------
    # Pointers
    # -------------------------------------------------
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    lse_ptr,
    qo_indptr_ptr,
    kv_indptr_ptr,
    # -------------------------------------------------
    # Scalars
    # -------------------------------------------------
    sm_scale,
    total_q,
    total_kv,
    # -------------------------------------------------
    # Strides
    # -------------------------------------------------
    stride_q_tok,
    stride_q_h,
    stride_q_d,
    stride_kv_tok,
    stride_kv_h,
    stride_kv_d,
    stride_out_tok,
    stride_out_h,
    stride_out_d,
    stride_lse_tok,
    stride_lse_h,
    # -------------------------------------------------
    # Compile‑time constants (filled by the autotuner)
    # -------------------------------------------------
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_QO_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GQA_RATIO: tl.constexpr,
):
    """
    Single‑kernel ragged‑batch GQA causal pre‑fill attention.
    Grid dimensions:
        program_id(0) – query‑tile (M dimension)
        program_id(1) – output head index (0 … NUM_QO_HEADS‑1)
        program_id(2) – batch index
    The inner KV‑loop loads each KV‑tile **once** and re‑uses it for
    all Q‑rows of the current tile.
    """
    # -------------------------------------------------
    # 0) Identify which batch we are processing
    # -------------------------------------------------
    batch_id = tl.program_id(2)

    q_start = tl.load(qo_indptr_ptr + batch_id)
    q_end = tl.load(qo_indptr_ptr + batch_id + 1)
    kv_start = tl.load(kv_indptr_ptr + batch_id)
    kv_end = tl.load(kv_indptr_ptr + batch_id + 1)

    num_q_tokens = q_end - q_start
    num_kv_tokens = kv_end - kv_start

    # -------------------------------------------------
    # 1) Program identifiers that stay constant for a block
    # -------------------------------------------------
    prog_id = tl.program_id(0)               # tile index in the query dimension
    head_idx = tl.program_id(1)              # output head index (0 … NUM_QO_HEADS‑1)

    kv_head = head_idx // GQA_RATIO           # which KV‑head this Q‑head uses
    delta = num_kv_tokens - num_q_tokens      # causal‑mask offset (can be negative)

    # -------------------------------------------------
    # 2) Grid‑stride loop over the query tiles
    # -------------------------------------------------
    stride_q_tiles = tl.num_programs(0)
    q_tile_start = prog_id * BLOCK_M

    while q_tile_start < num_q_tokens:
        # ------------------------------
        # 2️⃣ Load Q tile (BLOCK_M × BLOCK_D) – BF16
        # ------------------------------
        q_offs_m = tl.arange(0, BLOCK_M)
        q_offs_d = tl.arange(0, BLOCK_D)

        q_mask = (q_tile_start + q_offs_m[:, None] < num_q_tokens) & (q_offs_d[None, :] < BLOCK_D)

        q_ptrs = (
            q_ptr
            + (q_start + q_tile_start + q_offs_m[:, None]) * stride_q_tok
            + head_idx * stride_q_h
            + q_offs_d[None, :] * stride_q_d
        )
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)          # BF16 → stays BF16

        # ------------------------------
        # 3️⃣ Initialise stable‑softmax accumulators for this Q‑tile
        # ------------------------------
        m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)   # max per query
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                # Σexp
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)       # output accumulator

        # -------------------------------------------------
        # 4️⃣ Loop over KV tiles – each tile is loaded **once**
        # -------------------------------------------------
        # One‑pass early‑exit bound: for the current Q‑tile we never need KV
        # positions beyond max_kv_idx_tile, defined by the causal mask.
        q_tile_last = tl.minimum(q_tile_start + BLOCK_M - 1, num_q_tokens - 1)
        max_kv_idx_tile = tl.minimum(q_tile_last + 1 + delta, num_kv_tokens)

        kv_block_start = tl.zeros([], dtype=tl.int32)   # scalar 0
        while kv_block_start < max_kv_idx_tile:
            # ----- Load K (BF16) -------------------------------------------------
            k_offs_n = tl.arange(0, BLOCK_N)
            k_offs_d = tl.arange(0, BLOCK_D)

            k_mask = (kv_block_start + k_offs_n[:, None] < num_kv_tokens) & (
                k_offs_d[None, :] < BLOCK_D
            )
            k_ptrs = (
                k_ptr
                + (kv_start + kv_block_start + k_offs_n[:, None]) * stride_kv_tok
                + kv_head * stride_kv_h
                + k_offs_d[None, :] * stride_kv_d
            )
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)          # BF16

            # ----- Causal mask as additive bias ---------------------------------
            q_pos = q_tile_start + q_offs_m                               # absolute q indices
            max_kv_idx = q_pos + 1 + delta                               # inclusive upper bound per query
            kv_pos = kv_block_start + k_offs_n

            mask_bias = tl.where(
                kv_pos[None, :] < max_kv_idx[:, None],
                tl.zeros([], dtype=tl.float32),
                tl.full([], -1e9, dtype=tl.float32),
            )  # [BLOCK_M, BLOCK_N]

            # ----- Q·Kᵀ + bias (BF16×BF16 → FP32) ---------------------------------
            qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * sm_scale + mask_bias  # [BLOCK_M, BLOCK_N]

            # ----- Stable soft‑max update ------------------------------------
            m_i_new = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_i_new[:, None])                # un‑normalised probs (FP32)
            alpha = tl.exp(m_i - m_i_new)                    # correction factor (FP32)
            l_i_new = alpha * l_i + tl.sum(p, axis=1)

            # ----- Load V (BF16 → FP32) --------------------------------------
            v_offs_n = tl.arange(0, BLOCK_N)
            v_offs_d = tl.arange(0, BLOCK_D)

            v_mask = (kv_block_start + v_offs_n[:, None] < num_kv_tokens) & (
                v_offs_d[None, :] < BLOCK_D
            )
            v_ptrs = (
                v_ptr
                + (kv_start + kv_block_start + v_offs_n[:, None]) * stride_kv_tok
                + kv_head * stride_kv_h
                + v_offs_d[None, :] * stride_kv_d
            )
            v = tl.load(v_ptrs, mask=v_mask, other=0.0)        # BF16
            v_f32 = v.to(tl.float32)                         # convert once per tile

            # ----- Accumulate ------------------------------------------------
            acc = acc * alpha[:, None]                       # rescale previous contribution
            acc += tl.dot(p, v_f32, out_dtype=tl.float32)    # [BLOCK_M, BLOCK_D]

            # ----- Update running max / sum ---------------------------------
            m_i = m_i_new
            l_i = l_i_new

            # ----- advance KV block -------------------------------------------------
            kv_block_start += BLOCK_N

        # -------------------------------------------------
        # 5️⃣ Normalise and write outputs for this Q‑tile
        # -------------------------------------------------
        acc = acc / l_i[:, None]                     # [BLOCK_M, BLOCK_D]

        # ----- Write attention output ---------------------------------------
        out_offs_m = tl.arange(0, BLOCK_M)
        out_offs_d = tl.arange(0, BLOCK_D)
        out_mask = (q_tile_start + out_offs_m[:, None] < num_q_tokens) & (
            out_offs_d[None, :] < BLOCK_D
        )
        out_ptrs = (
            output_ptr
            + (q_start + q_tile_start + out_offs_m[:, None]) * stride_out_tok
            + head_idx * stride_out_h
            + out_offs_d[None, :] * stride_out_d
        )
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)

        # ----- Write LSE (log‑sum‑exp in base‑2) ---------------------------
        lse_offs = tl.arange(0, BLOCK_M)
        lse_mask = q_tile_start + lse_offs < num_q_tokens
        lse_ptrs = (
            lse_ptr
            + (q_start + q_tile_start + lse_offs) * stride_lse_tok
            + head_idx * stride_lse_h
        )
        log2_e = 1.4426950408889634      # 1 / ln(2)
        lse_val = (m_i + tl.log(l_i)) * log2_e
        tl.store(lse_ptrs, lse_val, mask=lse_mask)

        # -------------------------------------------------
        # 6️⃣ Advance to the next Q‑tile (grid‑stride)
        # -------------------------------------------------
        q_tile_start += stride_q_tiles * BLOCK_M


# -------------------------------------------------------------
# 2️⃣  Host‑side driver – grid size adapts to the autotuned BLOCK_M
# -------------------------------------------------------------
def run(q, k, v, qo_indptr, kv_indptr, sm_scale):
    """
    Triton implementation of ragged GQA causal pre‑fill attention
    (single‑kernel, fully autotuned, with additive‑bias causal masking
    and per‑KV‑tile reuse across Q‑tiles).
    """
    # ---------------------------------------------------------
    # 0) Make sure everything lives on CUDA
    # ---------------------------------------------------------
    orig_devices = [t.device for t in (q, k, v, qo_indptr, kv_indptr)]
    needs_cuda = [not t.is_cuda for t in (q, k, v, qo_indptr, kv_indptr)]

    if any(needs_cuda):
        if needs_cuda[0]:
            q = q.cuda()
        if needs_cuda[1]:
            k = k.cuda()
        if needs_cuda[2]:
            v = v.cuda()
        if needs_cuda[3]:
            qo_indptr = qo_indptr.cuda()
        if needs_cuda[4]:
            kv_indptr = kv_indptr.cuda()

    device = q.device

    # ---------------------------------------------------------
    # 1) Extract dimensions & sanity‑check
    # ---------------------------------------------------------
    total_q, num_qo_heads, head_dim = q.shape
    total_kv, num_kv_heads, _ = k.shape
    len_indptr = qo_indptr.shape[0]

    assert num_qo_heads == 32, "num_qo_heads must be 32"
    assert num_kv_heads == 4,  "num_kv_heads must be 4"
    assert head_dim == 128,    "head_dim must be 128"

    assert total_q == qo_indptr[-1].item()
    assert total_kv == kv_indptr[-1].item()

    GQA_RATIO = num_qo_heads // num_kv_heads

    # ---------------------------------------------------------
    # 2) Allocate outputs
    # ---------------------------------------------------------
    output = torch.zeros(
        (total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=device
    )
    lse = torch.full(
        (total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
    )

    # ---------------------------------------------------------
    # 3) Grid configuration – meta‑aware (depends on the chosen BLOCK_M)
    # ---------------------------------------------------------
    max_q_len = (qo_indptr[1:] - qo_indptr[:-1]).max().item()

    # Lambda receives the concrete meta‑dictionary after autotuning.
    grid = lambda META: (
        triton.cdiv(max_q_len, META["BLOCK_M"]),   # dim‑0 : query‑tiles
        num_qo_heads,                             # dim‑1 : heads
        len_indptr - 1,                           # dim‑2 : batches
    )

    # ---------------------------------------------------------
    # 4) Launch the kernel
    # ---------------------------------------------------------
    gqa_ragged_prefill_causal_kernel[grid](
        # pointers
        q,
        k,
        v,
        output,
        lse,
        qo_indptr,
        kv_indptr,
        # scalars
        sm_scale,
        total_q,
        total_kv,
        # strides
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse.stride(0),
        lse.stride(1),
        # compile‑time constants (autotuner will fill the rest)
        BLOCK_D=head_dim,
        NUM_QO_HEADS=num_qo_heads,
        NUM_KV_HEADS=num_kv_heads,
        GQA_RATIO=GQA_RATIO,
    )

    # ---------------------------------------------------------
    # 5) Move results back to the original device if needed
    # ---------------------------------------------------------
    if not orig_devices[0].type == "cuda":   # q was originally on CPU
        output = output.to(orig_devices[0])
        lse = lse.to(orig_devices[0])

    return output, lse