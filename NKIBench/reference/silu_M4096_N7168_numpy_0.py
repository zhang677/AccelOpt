
import numpy as np

M = 4096
N = 7168

def get_inputs():
    x = np.random.normal(loc=0, scale=1.0, size=(M, N)).astype(np.float32)
    return [x]
    

def forward(x):
  return x / (1 + np.exp(-x))


def transform_to_nki_inputs(inputs):
    tensor_inputs = []
    tensor_inputs.append(np.reshape(inputs[0], (128, 32, 7168)))  # input[0] -> tensor_input[0]

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
