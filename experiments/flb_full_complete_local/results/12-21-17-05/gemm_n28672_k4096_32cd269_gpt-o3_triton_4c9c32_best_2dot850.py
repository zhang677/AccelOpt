import torch
import triton
import triton.language as tl

# --------------------------------------------------------------
# GEMM kernel – unchanged (already stride‑aware)
# --------------------------------------------------------------
@triton.jit
def gemm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,   # stride for the logical “N” dimension of B
    stride_bk,   # stride for the logical “K” dimension of B
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """C = A @ B.T   (A: [M, K], B: [N, K], C: [M, N])"""
    pid_m = tl.program_id(0)          # row‑tile index
    pid_n = tl.program_id(1)          # col‑tile index

    # ----------------------------------------------------------
    # Tile offsets
    # ----------------------------------------------------------
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                     # [BLOCK_K]

    # Pointers to the current tile of A and B
    a_ptrs = A_ptr + (offs_m[:, None] * stride_am +
                     offs_k[None, :] * stride_ak)      # [BLOCK_M, BLOCK_K]
    b_ptrs = B_ptr + (offs_n[None, :] * stride_bn +
                     offs_k[:, None] * stride_bk)      # [BLOCK_K, BLOCK_N]

    # Accumulator in fp32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ----------------------------------------------------------
    # Main K‑loop (tiling over the reduction dimension)
    # ----------------------------------------------------------
    num_k_iters = tl.cdiv(K, BLOCK_K)
    for _ in range(num_k_iters):
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < K),
            other=0.0,
        )
        acc += tl.dot(a, b)

        # advance to the next K‑tile
        offs_k += BLOCK_K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # ----------------------------------------------------------
    # Write back result
    # ----------------------------------------------------------
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm +
                      offs_n[None, :] * stride_cn)
    tl.store(
        c_ptrs,
        acc.to(tl.float16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


# --------------------------------------------------------------
# Host‑side launcher – **no transpose of B any more**
# --------------------------------------------------------------
def run(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    High‑performance GEMM for the fixed shape:
        A : [M, 4096]   (float16)
        B : [28672, 4096] (float16)   <-- **used directly**, no copy

    Returns C : [M, 28672] (float16)
    """
    # ------------------------------------------------------------------
    # sanity checks and device handling
    # ------------------------------------------------------------------
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2‑D tensors")
    if A.shape[1] != 4096 or B.shape[1] != 4096 or B.shape[0] != 28672:
        raise ValueError("Expected shapes: A [M, 4096], B [28672, 4096]")
    if A.dtype != torch.float16 or B.dtype != torch.float16:
        raise TypeError("A and B must be torch.float16")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this Triton kernel")

    device = torch.device("cuda")
    A = A.to(device) if not A.is_cuda else A
    B = B.to(device) if not B.is_cuda else B

    # ------------------------------------------------------------------
    # No one‑time transpose; we just pass B with its native strides
    # ------------------------------------------------------------------
    M = A.shape[0]
    N = 28672            # fixed by the problem definition
    K = 4096

    # Output tensor
    C = torch.empty((M, N), device=device, dtype=torch.float16)

    # --------------------------------------------------------------
    # Tile configuration (same as baseline – empirically fast)
    # --------------------------------------------------------------
    BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64
    NUM_WARPS, NUM_STAGES = 8, 4

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # --------------------------------------------------------------
    # Launch the kernel with the original B and its strides
    # --------------------------------------------------------------
    gemm_kernel[grid](
        A,                         # [M, K]
        B,                         # original layout [N, K]
        C,
        M, N, K,
        A.stride(0), A.stride(1),          # stride_am, stride_ak
        B.stride(0), B.stride(1),          # stride_bn = K (=4096), stride_bk = 1
        C.stride(0), C.stride(1),          # stride_cm, stride_cn
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    torch.cuda.synchronize()
    return C


# ------------------------------------------------------------------
# Simple correctness test (run when the file is executed directly)
# ------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    M_test = 256
    A_test = torch.randn((M_test, 4096), dtype=torch.float16, device="cuda")
    B_test = torch.randn((28672, 4096), dtype=torch.float16, device="cuda")

    # Reference: fp32 accumulation then cast back to fp16
    C_ref = (A_test.float() @ B_test.t().float()).half()

    C_out = run(A_test, B_test)

    assert torch.allclose(C_ref, C_out, atol=1e-2, rtol=1e-2)
    print("✅ correctness test passed")