import torch
import triton
import triton.language as tl
import math

# Reference implementation for correctness check
@torch.no_grad()
def reference_run(hidden_states, residual, weight):
    """
    Reference PyTorch implementation for fused_add_rmsnorm_h7168.
    """
    _, hidden_size = hidden_states.shape
    # Check constants
    assert hidden_size == 7168

    EPS = 1e-6

    x = hidden_states.to(torch.float32) + residual.to(torch.float32)
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    y = (x * inv_rms) * weight.to(torch.float32)
    return y.to(hidden_states.dtype)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 1024, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_SIZE_N': 1024, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_SIZE_N': 2048, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_SIZE_N': 4096, 'num_warps': 16, 'num_stages': 2}),
        # This config is likely optimal as it covers the entire row in one loop iteration
        triton.Config({'BLOCK_SIZE_N': 8192, 'num_warps': 16, 'num_stages': 2}),
    ],
    key=['HIDDEN_SIZE'],
)
@triton.jit
def _fused_add_rmsnorm_h7168_kernel(
    # Pointers to tensors
    hidden_states_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    # Stride information
    stride_hs_b,
    stride_res_b,
    stride_out_b,
    # Constants
    HIDDEN_SIZE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Triton kernel for Fused Add + RMSNorm.
    Each program instance processes one row of the input tensors.
    """
    # -----------------------------------------------------------
    # Grid and pointers
    # -----------------------------------------------------------
    
    # Each program instance computes a single row (batch element)
    pid_b = tl.program_id(axis=0)

    # Pointers to the start of the current row for each tensor
    row_hs_ptr = hidden_states_ptr + pid_b * stride_hs_b
    row_res_ptr = residual_ptr + pid_b * stride_res_b
    row_out_ptr = output_ptr + pid_b * stride_out_b
    
    # -----------------------------------------------------------
    # Pass 1: Compute sum of squares
    # -----------------------------------------------------------
    
    # Accumulator for the sum of squares, initialized to zeros
    acc = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    
    # Loop over the hidden dimension in blocks of BLOCK_SIZE_N
    for off in range(0, HIDDEN_SIZE, BLOCK_SIZE_N):
        offsets = off + tl.arange(0, BLOCK_SIZE_N)
        mask = offsets < HIDDEN_SIZE

        # Load input tensors `hidden_states` and `residual`
        hs = tl.load(row_hs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        res = tl.load(row_res_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Fused add
        x = hs + res
        
        # Square and accumulate
        acc += x * x
        
    # Reduce the accumulator to a single scalar value for the variance
    # tl.sum sums across all threads in a block
    variance = tl.sum(acc, axis=0) / HIDDEN_SIZE
    
    # Compute the inverse root mean square
    inv_rms = tl.rsqrt(variance + EPS)

    # -----------------------------------------------------------
    # Pass 2: Normalize and store
    # -----------------------------------------------------------
    
    # Loop over the hidden dimension again to apply the normalization
    for off in range(0, HIDDEN_SIZE, BLOCK_SIZE_N):
        offsets = off + tl.arange(0, BLOCK_SIZE_N)
        mask = offsets < HIDDEN_SIZE

        # Reload inputs
        hs = tl.load(row_hs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        res = tl.load(row_res_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        
        # Load weight
        w = tl.load(weight_ptr + offsets, mask=mask).to(tl.float32)
        
        # Fused add
        x = hs + res
        
        # Apply RMSNorm and scale by weight
        normalized_x = x * inv_rms
        output_val = normalized_x * w
        
        # Store the result
        tl.store(row_out_ptr + offsets, output_val.to(tl.bfloat16), mask=mask)


def run(*args, **kwargs):
    """
    Wrapper function to run the fused_add_rmsnorm_h7168 Triton kernel.

    Handles device management, tensor validation, grid computation, and kernel launch.
    Moves data to the GPU if necessary and returns the result on the original device.

    Args:
        hidden_states (torch.Tensor): Input tensor of shape [batch_size, 7168] and dtype bfloat16.
        residual (torch.Tensor): Residual tensor of shape [batch_size, 7168] and dtype bfloat16.
        weight (torch.Tensor): Weight tensor of shape [7168] and dtype bfloat16.

    Returns:
        torch.Tensor: The output tensor of the same shape and dtype as hidden_states.
    """
    # 1. Unpack arguments
    if args:
        hidden_states, residual, weight = args
    else:
        hidden_states = kwargs.get('hidden_states')
        residual = kwargs.get('residual')
        weight = kwargs.get('weight')
        if hidden_states is None or residual is None or weight is None:
            raise ValueError("Missing required arguments: 'hidden_states', 'residual', 'weight'")

    # 2. Device Management
    if not torch.cuda.is_available():
        raise RuntimeError("Triton requires a CUDA-enabled GPU, but CUDA is not available.")
    
    # Use the device of the first input tensor as the primary execution device
    # If the first tensor is on CPU, move all to the default CUDA device
    primary_input_device = hidden_states.device
    if primary_input_device.type == 'cpu':
        execution_device = torch.device('cuda')
    else:
        execution_device = primary_input_device

    # Move all tensors to the execution device
    hidden_states_gpu = hidden_states.to(execution_device)
    residual_gpu = residual.to(execution_device)
    weight_gpu = weight.to(execution_device)

    # 3. Shape and DType Validation
    batch_size, hidden_size = hidden_states.shape
    
    if hidden_size != 7168:
        raise ValueError(f"hidden_size must be 7168, but got {hidden_size}")
    if residual.shape != hidden_states.shape:
        raise ValueError(f"Shape of residual {residual.shape} does not match hidden_states {hidden_states.shape}")
    if weight.shape != (hidden_size,):
        raise ValueError(f"Shape of weight {weight.shape} does not match expected ({hidden_size},)")

    expected_dtype = torch.bfloat16
    if hidden_states.dtype != expected_dtype or residual.dtype != expected_dtype or weight.dtype != expected_dtype:
        raise TypeError(f"All input tensors must have dtype {expected_dtype}")

    # 4. Allocate Output Tensor
    output_gpu = torch.empty_like(hidden_states_gpu)

    # 5. Grid Computation
    # The grid is 1D, with one program instance per batch element.
    grid = (batch_size,)

    # 6. Kernel Launch
    _fused_add_rmsnorm_h7168_kernel[grid](
        hidden_states_gpu,
        residual_gpu,
        weight_gpu,
        output_gpu,
        hidden_states_gpu.stride(0),
        residual_gpu.stride(0),
        output_gpu.stride(0),
        HIDDEN_SIZE=hidden_size,
        EPS=1e-6,
    )

    # 7. Restore Original Device
    # Move the output tensor back to the device of the original `hidden_states` tensor
    output = output_gpu.to(primary_input_device)

    return output