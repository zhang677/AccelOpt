import torch
from functools import reduce
import math
from torch.utils.module_tracker import ModuleTracker
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import shape_wrapper
from collections import defaultdict
from typing import Dict, Any
import operator

def product(iterable):
    return reduce(operator.mul, iterable, 1)

def summation(iterable):
    return reduce(operator.add, iterable, 0)

class FlopCounterMode(TorchDispatchMode):
    def __init__(
            self,
            custom_mapping: Dict[Any, Any],
            depth: int = 2):
        super().__init__()
        self.flop_counts: Dict[str, Dict[Any, Any]]  = defaultdict(lambda: defaultdict(lambda: (0, 0)))
        self.depth = depth
        self.flop_registry ={
            **{k: v if getattr(v, "_get_raw", False) else shape_wrapper(v) for k, v in custom_mapping.items()}
        }
        self.mod_tracker = ModuleTracker()
        self.unsupported_ops = set()
    
    def get_total_flops(self):
        values = self.flop_counts['Global'].values()
        total_flops = sum([v[0] for v in values])
        total_mem = sum([v[1] for v in values])
        return (total_flops, total_mem)
    
    def get_flop_counts(self):
        return {k: dict(v) for k, v in self.flop_counts.items()}

    def get_unsupported_ops(self):
        return self.unsupported_ops
    
    def __enter__(self):
        self.flop_counts.clear()
        self.mod_tracker.__enter__()
        super().__enter__()
        return self

    def __exit__(self, *args):
        super().__exit__(*args)
        self.mod_tracker.__exit__()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}
        out = func(*args, **kwargs)
        return self._count_flops(func._overloadpacket, out, args, kwargs)

    def _count_flops(self, func_packet, out, args, kwargs):
        if func_packet in self.flop_registry:
            flop_count_func = self.flop_registry[func_packet]
            flop_count = flop_count_func(*args, **kwargs, out_val=out)  # type: ignore[operator]
            for par in set(self.mod_tracker.parents):
                if isinstance(flop_count, tuple):
                    self.flop_counts[par][func_packet] = (self.flop_counts[par][func_packet][0] + flop_count[0],
                                                            self.flop_counts[par][func_packet][1] + flop_count[1])
                elif isinstance(flop_count, int):
                    self.flop_counts[par][func_packet] += flop_count
                else:
                    raise TypeError(f"Unsupported flop count type: {type(flop_count)}")
        else:
            self.unsupported_ops.add(func_packet)
        return out

def get_binary_op_shape(a_shape, b_shape, out_shape=None):
    """Get the shape for binary operations (like addition or multiplication).
    
    This function handles broadcasting rules to determine the output shape.
    """
    if out_shape is None:
        # If we don't have the output shape, we can calculate it
        # through broadcasting rules
        shape = torch.broadcast_shapes(a_shape, b_shape)
    else:
        shape = out_shape
    return shape

# Create FLOPs counting functions for elementwise operations
def flop_binary(a_shape, b_shape, out_shape=None, **kwargs):
    if not isinstance(a_shape, torch.Size) and not isinstance(b_shape, torch.Size):
        shape = 1
    elif isinstance(b_shape, torch.Size) and not isinstance(a_shape, torch.Size):
        shape = b_shape
    elif isinstance(a_shape, torch.Size) and not isinstance(b_shape, torch.Size):
        shape = a_shape
    else:
        shape = get_binary_op_shape(a_shape, b_shape, out_shape)
        # One operation per element
    return (product(shape), 3 * product(shape)) 

def flop_zero(a_shape, *args, out_shape=None, **kwargs):
    return (0, summation((product(a_shape), product(out_shape))))

def flop_matmul(a_shape, b_shape, out_shape=None, **kwargs):
    # Assuming a_shape is (M, K) and b_shape is (K, N)
    # The output shape will be (M, N)
    k_a = a_shape[-1]
    k_b = b_shape[0]
    assert k_a == k_b, "Inner dimensions must match for matrix multiplication"
    return (2 * product(a_shape[:-1]) * product(b_shape[1:]) * k_a,
            summation((product(a_shape), product(b_shape), product(out_shape))))

def flop_softmax(a_shape, dim, _stacklevel, out_shape=None, **kwargs):
    return (3 * product(a_shape),  # exp + sum + division
            summation((product(a_shape), product(out_shape))))

def flop_topk(input_shape, k, dim=None, largest=True, sorted=True, out_shape=None, **kwargs):
    # Print arguments for debugging
    # print(f"input_shape: {input_shape}, k: {k}, dim: {dim}, largest: {largest}, sorted: {sorted}, out_shape: {out_shape}")
    # If dim is not specified, it defaults to the last dimension
    if dim is None:
        dim = len(input_shape) - 1
    
    # Ensure dim is positive for easier handling
    if dim < 0:
        dim = len(input_shape) + dim
    
    # Total number of elements in the tensor
    total_elements = reduce(lambda x, y: x * y, input_shape, 1)
    
    # Number of elements along the specified dimension
    dim_size = input_shape[dim]
    
    # Number of slices across the specified dimension
    num_slices = total_elements // dim_size
    
    # FLOPs for sorting: O(n log n) for each slice
    # We use n log n comparison operations
    # Each comparison involves ~1 FLOP
    sorting_flops = num_slices * dim_size * math.log2(dim_size)
    
    # FLOPs for selection: O(n) for each slice to select top k elements
    # If sorted=True, we need an additional k*log(k) operations
    selection_flops = num_slices * dim_size
    
    if sorted and k > 1:
        # Additional cost for sorting the top k elements
        sorting_k_flops = num_slices * k * math.log2(k)
    else:
        sorting_k_flops = 0
    
    # Total FLOPs estimation
    return (int(sorting_flops + sorting_k_flops),
            summation((int(selection_flops), product(input_shape), product(out_shape[0]), product(out_shape[1]))))

def flop_scatter_(input_shape, dim, index_shape, value, out_shape=None, **kwargs):
    """
    Calculate the number of floating-point operations for torch.scatter_ operation.
    
    Args:
        input_shape (tuple): Shape of the destination tensor that is modified in-place.
        dim (int): Dimension along which to index.
        index_shape (tuple): Shape of the index tensor.
        src_shape (tuple, optional): Shape of the source tensor. If None, assumed to be same as index_shape.
        reduce (str, optional): Reduction operation to perform: None, 'add', 'multiply'. Default is None.
        
    Returns:
        int: Estimated number of FLOPs.
    """    
    # Calculate the total number of elements in the index tensor
    # This equals the number of scatter operations
    total_index_elements = reduce(lambda x, y: x * y, index_shape, 1)
    
    # Basic operation: read from source and write to destination (2 memory operations)
    # This doesn't actually involve floating point operations in the strict sense
    memory_ops = 2 * total_index_elements
    
    # Total FLOPs estimation
    # Note: Memory operations aren't technically FLOPs, but we include them in the count
    # to represent the computational cost
    return (0, memory_ops)

def flop_mm(a_shape, b_shape, *args, out_shape=None, **kwargs):
    """Count flops for matmul."""
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two matrices.
    m, k = a_shape
    k2, n = b_shape
    assert k == k2
    return (product((m, n, 2, k)),
            summation((product(a_shape), product(b_shape), product(out_shape))))

def flop_unary(a_shape, *args, out_shape=None, **kwargs):
    """Count flops for unary operations."""
    return (product(a_shape), summation((product(a_shape), product(out_shape))))

def flop_mean(a_shape, *args, out_shape=None, **kwargs):
    """Count flops for mean operations."""
    return (product(a_shape) + product(out_shape), summation((product(a_shape), product(out_shape))))

def flop_stack(*args, out_shape=None, **kwargs):
    """Count flops for stack operations."""
    # Print arguments for debugging
    return (0, 2 * product(out_shape))

def flop_pow(base_shape, exp_shape, out_shape=None, **kwargs):
    if not isinstance(base_shape, torch.Size) and isinstance(exp_shape, torch.Size):
        shape = exp_shape
    elif isinstance(base_shape, torch.Size) and not isinstance(exp_shape, torch.Size):
        shape = base_shape
    else:
        return (product(out_shape), summation((product(base_shape), product(exp_shape), product(out_shape))))
    return (product(shape), summation((product(shape), product(out_shape))))

def flop_bmm(a_shape, b_shape, out_shape=None, **kwargs):
    """Count flops for the bmm operation."""
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two tensor.
    b, m, k = a_shape
    b2, k2, n = b_shape
    assert b == b2
    assert k == k2
    # NB(chilli): Should be 2 * k - 1 technically for FLOPs.
    flop = b * m * n * 2 * k
    return (flop, summation((product(a_shape), product(b_shape), product(out_shape))))

def flop_arange(*args, out_shape=None, **kwargs):
    return (0, product(out_shape))

def flop_sigmoid(*args, out_shape=None, **kwargs):
    """Count flops for sigmoid operation."""
    # Sigmoid is a simple function, we can assume 1 FLOP per element
    return (3 * product(out_shape), 2 * product(out_shape))

def flop_gelu(*args, out_shape=None, **kwargs):
    """Count flops for GELU operation."""
    # GELU(x)=0.5 * x * (1+Tanh(2/π * (x+0.044715 * x^3)))
    return (14 * product(out_shape), 2 * product(out_shape))

def flop_var(a_shape, *args, out_shape=None, **kwargs):
    """Count flops for variance operation."""
    # Variance is computed as:
    # Var(X) = E[X^2] - (E[X])^2
    # Assuming we compute E[X] and E[X^2] separately
    return (3 * product(a_shape), summation((product(a_shape), product(out_shape))))

def flop_mv(a_shape, b_shape, out_shape=None, **kwargs):
    """Count flops for matrix-vector multiplication."""
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two matrices.
    m, k = a_shape
    k2 = b_shape[0]
    assert k == k2
    return (m * k * 2, summation((product(a_shape), product(b_shape), product(out_shape))))

def flop_dot(a_shape, b_shape, out_shape=None, **kwargs):
    """Count flops for dot product."""
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two vectors.
    assert len(a_shape) == 1 and len(b_shape) == 1
    assert a_shape[0] == b_shape[0]
    return (a_shape[0] * 2, summation((product(a_shape), product(b_shape), product(out_shape))))

def flop_split(a_shape, *args, out_shape=None, **kwargs):
    """Count flops for split operation."""
    # Split does not involve any floating-point operations
    # but we can count the memory operations
    return (0, 2 * product(a_shape))

custom_flops_mapping = {
    torch.ops.aten.mm: flop_mm,
    torch.ops.aten.bmm: flop_bmm,
    torch.ops.aten.add: flop_binary,
    torch.ops.aten.mul: flop_binary,
    torch.ops.aten.sub: flop_binary,
    torch.ops.aten.rsub: flop_binary,
    torch.ops.aten.div: flop_binary,
    torch.ops.aten.max: flop_zero,
    torch.ops.aten.exp: flop_unary,
    torch.ops.aten._softmax: flop_softmax,
    torch.ops.aten.topk: flop_topk,
    torch.ops.aten.scatter_: flop_scatter_,
    torch.ops.aten.scatter: flop_scatter_,
    torch.ops.aten.pow: flop_pow,
    torch.ops.aten.sqrt: flop_unary,
    torch.ops.aten.rsqrt: flop_unary,
    torch.ops.aten.mean: flop_mean,
    torch.ops.aten.expand: flop_zero,
    torch.ops.aten.stack: flop_stack,
    torch.ops.aten.permute: flop_zero,
    torch.ops.aten.transpose: flop_zero,
    torch.ops.aten.sum: flop_unary,
    torch.ops.aten.reciprocal: flop_unary,
    torch.ops.aten.sin: flop_unary,
    torch.ops.aten.cos: flop_unary,
    torch.ops.aten.copy_: flop_zero,
    torch.ops.aten.arange: flop_arange,
    torch.ops.aten.sigmoid: flop_sigmoid,
    torch.ops.aten.gelu: flop_gelu,
    torch.ops.aten.silu: flop_sigmoid,
    torch.ops.aten.var: flop_var,
    torch.ops.aten.mv: flop_mv,
    torch.ops.aten.dot: flop_dot,
    torch.ops.aten.maximum: flop_binary,
    torch.ops.aten.cat: flop_stack,
    torch.ops.aten.split: flop_split
}


def count_flops(model: torch.nn.Module, input_args: tuple):
    with FlopCounterMode(custom_flops_mapping) as flop_mode:
        model(*input_args)
        flop_counts = flop_mode.get_flop_counts()
        if 'Global' in flop_counts:
            flop_count = flop_counts['Global']
            summary_count = flop_mode.get_total_flops()
        else:
            flop_count = [0, 0]
            summary_count = [0, 0]
        unsupported_ops = flop_mode.get_unsupported_ops()

    return {
        "flops": flop_count, 
        "summary": summary_count,
        "unsup_ops": list(unsupported_ops),
    }

def post_process_ops(data):
    # Process flops: convert OpOverloadPacket keys to their op string values
    processed_flops = {}
    if not isinstance(data['flops'], dict):
        processed_flops = data['flops']
    else:
        for op_packet, value in data['flops'].items():
            # Extract the op name from the OpOverloadPacket
            op_name = str(op_packet)
            processed_flops[op_name] = list(value)
    
    # Process unsupported ops: extract op strings from OpOverloadPacket objects
    processed_unsup_ops = [str(op_packet) for op_packet in data['unsup_ops']]
    
    # Return the processed data
    return {
        'flops': processed_flops,
        'summary': list(data['summary']),
        'unsup_ops': processed_unsup_ops
    }