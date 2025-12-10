
import numpy as np

M = 4096
N = 2048
K = 1024

def get_inputs():
    x = np.random.normal(loc=0, scale=1.0, size=(M, K)).astype(np.float32)
    w = np.random.normal(loc=0, scale=1.0, size=(K, N)).astype(np.float32)
    return [x, w]
    

def forward(input_tensor, weight_matrix):
  # RMSNorm calculations
  squared_input = np.square(input_tensor)
  scaled_square = np.divide(squared_input, K)

  rms_sum = np.sum(scaled_square, axis=1, keepdims=True)
  rms_norm = np.sqrt(rms_sum)

  normalized = np.divide(input_tensor, rms_norm)

  # Matrix multiplication
  matmul_result = np.matmul(normalized, weight_matrix)

  return matmul_result



def transform_to_nki_inputs(inputs):
    tensor_inputs = []
    tensor_inputs.append(np.reshape(inputs[0], (32, 128, 1024)))  # input[0] -> tensor_input[0]
    tensor_inputs.append(np.reshape(inputs[1], (8, 128, 2048)))  # input[1] -> tensor_input[1]

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
