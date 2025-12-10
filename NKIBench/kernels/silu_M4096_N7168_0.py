import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim

@nki.jit
def kernel(v1):
    import numpy as np
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.typing as nt
    import neuronxcc.nki.isa as nisa
    from neuronxcc.nki import trace
    from neuronxcc.nki.language import par_dim
    v2 = nl.ndarray((128, 32, 7168), dtype=np.float32, buffer=nl.shared_hbm)
    v3 = nl.ndarray((nl.par_dim(128), 1), dtype=np.float32, name='memset.172', buffer=nl.sbuf)
    v4 = nl.ndarray((32, 7, nl.par_dim(128), 1024), dtype=np.float32, name='x_local_152', buffer=nl.sbuf)
    v5 = nl.ndarray((32, 7, nl.par_dim(128), 1024), dtype=np.float32, name='', buffer=nl.sbuf)
    v6 = nl.ndarray((32, 7, 2, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v7 = nl.ndarray((32, 7, 2, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v8 = nl.ndarray((32, 7, 2, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v3[nl.arange(128)[:, None], 0] = nisa.memset(shape=(128, 1), value=np.dtype(np.uint16).type(0), dtype=np.float32, mask=None, engine=nki.isa.unknown_engine)
    for i0 in nl.affine_range(32):
        for i1 in nl.affine_range(7):
            v4[i0, i1, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nl.load(v1[nl.arange(128)[:, None], i0, nl.arange(1024)[None, :] + 1024 * i1], dtype=np.float32, mask=None)
            for i2 in nl.affine_range(2):
                v6[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.activation(op=nl.exp, data=v4[i0, i1, nl.arange(128)[:, None], 512 * i2 + nl.arange(512)[None, :]], bias=v3[nl.arange(128)[:, None], 0], scale=-1.0, mask=None, dtype=np.float32)
                v7[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.tensor_scalar(data=v6[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]], op0=nl.add, operand0=np.dtype(np.float32).type(1), reverse0=True, dtype=np.float32, mask=None, engine=nki.isa.unknown_engine)
                v8[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.reciprocal(data=v7[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=None, dtype=np.float32)
                v5[i0, i1, nl.arange(128)[:, None], 512 * i2 + nl.arange(512)[None, :]] = nl.multiply(v4[i0, i1, nl.arange(128)[:, None], 512 * i2 + nl.arange(512)[None, :]], v8[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=None, dtype=np.float32)
                ' end loop i2 '
            nl.store(v2[nl.arange(128)[:, None], i0, nl.arange(1024)[None, :] + 1024 * i1], value=v5[i0, i1, nl.arange(128)[:, None], nl.arange(1024)[None, :]], mask=None)
            ' end loop i1 '
        ' end loop i0 '
    return v2