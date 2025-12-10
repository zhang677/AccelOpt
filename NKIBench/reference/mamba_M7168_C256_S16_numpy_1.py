
import numpy as np

M = 7168
C = 256
S = 16

def get_inputs():
    delta = np.random.normal(loc=0, scale=0.05, size=(C, M)).astype(np.float32)
    u = np.random.normal(loc=0, scale=0.05, size=(C, M)).astype(np.float32)
    a = np.random.normal(loc=0, scale=0.05, size=(C, S)).astype(np.float32)
    b = np.random.normal(loc=0, scale=0.05, size=(S, M)).astype(np.float32)
    c = np.random.normal(loc=0, scale=0.05, size=(S, M)).astype(np.float32)
    return [delta, u, a, b, c]
    

def forward(delta, u, a, b, c):
  deltaA = np.exp(delta[:, None, :] * a[:, :, None])
  deltaB_u = delta[:, None, :] * b[None, :, :] * u[:, None, :]
  scan_res = np.ndarray((C, S, M), dtype=np.float32)
  for i in range(M):
      prev_state = scan_res[..., i - 1] if i > 0 else 0
      scan_res[..., i] = deltaA[..., i] * prev_state + deltaB_u[..., i]
  out = np.sum(c[None, :, :] * scan_res, axis=-2)
  return out

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
