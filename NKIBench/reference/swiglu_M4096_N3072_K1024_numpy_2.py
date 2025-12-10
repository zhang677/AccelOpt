
import numpy as np

M = 4096
N = 3072
K = 1024

def get_inputs():
    x = np.random.normal(loc=0, scale=1.0, size=(M, K)).astype(np.float32)
    w_up = np.random.normal(loc=0, scale=1.0, size=(K, N)).astype(np.float32)
    w_down = np.random.normal(loc=0, scale=1.0, size=(N, K)).astype(np.float32)
    w_gate = np.random.normal(loc=0, scale=1.0, size=(K, N)).astype(np.float32)
    return [x, w_up, w_down, w_gate]
    

def forward(x, w_up, w_down, w_gate):
  up_feature = np.matmul(x, w_up)
  gate_feature = np.matmul(x, w_gate)
  activated_gate_feature = gate_feature / (1 + np.exp(-gate_feature))
  return np.matmul(activated_gate_feature * up_feature, w_down)


def transform_to_nki_inputs(inputs):
    tensor_inputs = []
    tensor_inputs.append(np.reshape(inputs[0], (8, 4, 128, 8, 128)))  # input[0] -> tensor_input[0]
    tensor_inputs.append(np.reshape(inputs[1], (8, 128, 3072)))  # input[1] -> tensor_input[1]
    tensor_inputs.append(np.reshape(inputs[2], (24, 128, 1024)))  # input[2] -> tensor_input[2]
    tensor_inputs.append(np.reshape(inputs[3], (8, 128, 3072)))  # input[3] -> tensor_input[3]

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
