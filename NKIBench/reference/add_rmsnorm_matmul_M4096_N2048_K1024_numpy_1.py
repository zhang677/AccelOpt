
import numpy as np

M = 4096
N = 2048
K = 1024

def get_inputs():
  x = np.random.normal(loc=0, scale=1.0, size=(M, K)).astype(np.float32)
  w = np.random.normal(loc=0, scale=1.0, size=(K, N)).astype(np.float32)
  eps = 1e-5
  z = np.random.normal(loc=0, scale=1.0, size=(M, K)).astype(np.float32)
  g = np.random.normal(loc=0, scale=1.0, size=(K,)).astype(np.float32)
  return [x, w, eps, z, g]
    

def forward(x, w, eps, z, g):
  y = x + z
  t = np.square(y)
  t = np.divide(t, y.shape[-1])
  t = np.sum(t, axis=-1, keepdims=True)
  t = (t + eps).astype(y.dtype)
  y = y / np.sqrt(t)
  y = y * g
  return np.matmul(y, w)



def transform_to_nki_inputs(inputs):
    return inputs



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
