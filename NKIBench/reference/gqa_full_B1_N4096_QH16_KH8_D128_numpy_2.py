
import numpy as np

B = 1
N = 4096
QH = 16
KH = 8
D = 128

def get_inputs():
    q = np.random.normal(loc=0, scale=1.0, size=(B, N, QH, D)).astype(np.float32)
    k = np.random.normal(loc=0, scale=1.0, size=(B, N, KH, D)).astype(np.float32)
    v = np.random.normal(loc=0, scale=1.0, size=(B, N, KH, D)).astype(np.float32)
    return [q, k, v]
    

def forward(q, k, v):
  n_rep = QH // KH
  xk = np.repeat(k, n_rep, axis=2)
  xv = np.repeat(v, n_rep, axis=2)
  xq = q.transpose(0, 2, 1, 3)
  xk = xk.transpose(0, 2, 1, 3)
  xv = xv.transpose(0, 2, 1, 3)

  attention = (xq @ xk.transpose(0, 1, 3, 2)) / np.float32(np.sqrt(D))
  exp_attention = np.exp(attention - np.max(attention, axis=-1, keepdims=True))
  attention = exp_attention / np.sum(exp_attention, axis=-1, keepdims=True)

  output = attention @ xv
  return output



def transform_to_nki_inputs(inputs):
    tensor_inputs = []
    tensor_inputs.append(np.reshape(inputs[0], (32, 128, 16, 128)))  # input[0] -> tensor_input[0]
    tensor_inputs.append(np.reshape(inputs[1], (1, 8, 4, 128, 8, 128)))  # input[1] -> tensor_input[1]
    tensor_inputs.append(np.reshape(inputs[2], (1, 32, 128, 1024)))  # input[2] -> tensor_input[2]

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
