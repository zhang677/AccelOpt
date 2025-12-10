
import numpy as np

M = 4096
N = 10944
K = 2048

def get_inputs():
    lhs = np.random.normal(loc=0, scale=1.0, size=(K, M)).astype(np.float32)
    rhs = np.random.normal(loc=0, scale=1.0, size=(K, N)).astype(np.float32)
    return [lhs, rhs]
    

def forward(lhs, rhs):
  lhs_t = np.transpose(lhs, axes=(1, 0))
  return np.matmul(lhs_t, rhs)


def transform_to_nki_inputs(inputs):
    tensor_inputs = []
    tensor_inputs.append(np.reshape(inputs[0], (128, 16, 4096)))  # input[0] -> tensor_input[0]
    tensor_inputs.append(np.reshape(inputs[1], (128, 16, 10944)))  # input[1] -> tensor_input[1]

    return tensor_inputs



def transform_nki_outputs(k_res, ref):
    # Ensure outputs are in tuple form
    if not isinstance(k_res, tuple):
        k_res = (k_res,)
    
    refs = ref if isinstance(ref, tuple) else (ref,)
    k_outs = []
    
    for v, r in zip(k_res, refs):
        if hasattr(r, "shape"):
            k_outs.append(np.reshape(v, r.shape))
        else:
            k_outs.append(v)
    
    return k_outs
