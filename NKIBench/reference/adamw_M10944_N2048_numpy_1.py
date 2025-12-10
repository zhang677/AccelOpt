
import numpy as np

M = 10944
N = 2048

def get_inputs():
    theta = np.random.normal(loc=0, scale=1.0, size=(M, N)).astype(np.float32)
    g = np.random.normal(loc=0, scale=1.0, size=(M, N)).astype(np.float32)
    m = np.random.normal(loc=0, scale=1.0, size=(M, N)).astype(np.float32)
    v = np.abs(np.random.normal(loc=0, scale=1.0, size=(M, N))).astype(np.float32)
    return [theta, g, m, v]
    

def forward(theta, g, m, v):
  theta_t = theta - 1e-5 * theta
  m_t = 0.9 * m + 0.1 * g
  v_t = 0.999 * v + 0.001 * g * g
  v_hat = v_t * 1000
  new_theta_t = theta_t - 0.01 * m_t / (np.sqrt(v_hat) + 1e-8)
  return new_theta_t


def transform_to_nki_inputs(inputs):
    tensor_inputs = []
    tensor_inputs.append(np.reshape(inputs[0], (10944, 2048)))  # input[0] -> tensor_input[0]
    tensor_inputs.append(np.reshape(inputs[1], (10944, 2048)))  # input[1] -> tensor_input[1]
    tensor_inputs.append(np.reshape(inputs[2], (10944, 2048)))  # input[2] -> tensor_input[2]
    tensor_inputs.append(np.reshape(inputs[3], (10944, 2048)))  # input[3] -> tensor_input[3]

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
