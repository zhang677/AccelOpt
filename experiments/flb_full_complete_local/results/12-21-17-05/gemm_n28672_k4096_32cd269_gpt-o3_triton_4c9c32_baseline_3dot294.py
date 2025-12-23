import math
import torch
import triton
import triton.language as tl


@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)      # program id for M dimension
    pid_n = tl.program_id(1)      # program id for N dimension

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)              # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)              # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                                # [BLOCK_K]

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am +
                      offs_k[None, :] * stride_ak)                # [BLOCK_M, BLOCK_K]
    b_ptrs = B_ptr + (offs_n[None, :] * stride_bn +
                      offs_k[:, None] * stride_bk)                # [BLOCK_K, BLOCK_N]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_iters = tl.cdiv(K, BLOCK_K)
    for _ in range(num_k_iters):
        a = tl.load(
            a_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < K),
            other=0.0
        )
        acc += tl.dot(a, b)

        offs_k += BLOCK_K
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm +
                      offs_n[None, :] * stride_cn)
    acc = acc.to(tl.float16)
    tl.store(
        c_ptrs,
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )


def run(A: torch.Tensor, B: torch.Tensor):
    """
    High-performance GEMM on B200 GPUs.
    C = A @ B.T
    Shapes:
        A: [M, 4096]   (float16)
        B: [28672, 4096] (float16)
        C: [M, 28672]  (float16)
    """
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("A and B must be 2-D tensors")
    if A.shape[1] != 4096 or B.shape[1] != 4096 or B.shape[0] != 28672:
        raise ValueError("Expected shapes: A [M, 4096], B [28672, 4096]")
    if A.dtype != torch.float16 or B.dtype != torch.float16:
        raise TypeError("A and B must be float16")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this Triton kernel")

    orig_device_A = A.device
    orig_device_B = B.device

    A_cuda = A.cuda() if not A.is_cuda else A
    B_cuda = B.cuda() if not B.is_cuda else B

    M = A_cuda.shape[0]
    N = 28672
    K = 4096

    C_cuda = torch.empty((M, N), device=A_cuda.device, dtype=torch.float16)

    BLOCK_M = 128
    BLOCK_N = 256
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    gemm_kernel[grid](
        A_cuda, B_cuda, C_cuda,
        M, N, K,
        A_cuda.stride(0), A_cuda.stride(1),
        B_cuda.stride(0), B_cuda.stride(1),
        C_cuda.stride(0), C_cuda.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8,
        num_stages=4
    )

    torch.cuda.synchronize()

    if orig_device_A.type == "cuda":
        return C_cuda
    return C_cuda.cpu()


# Allow module import without immediate execution
if __name__ == "__main__":
    # Simple correctness test
    M_test = 256
    A_test = torch.randn((M_test, 4096), dtype=torch.float16)
    B_test = torch.randn((28672, 4096), dtype=torch.float16)
    C_ref = (A_test.float() @ B_test.t().float()).half()
    C_out = run(A_test, B_test)
    assert torch.allclose(C_ref, C_out, atol=1e-2, rtol=1e-2)
    print("Test passed!")