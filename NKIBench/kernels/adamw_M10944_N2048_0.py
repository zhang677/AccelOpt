import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim

@nki.jit
def kernel(v1, v2, v3, v4):
    import numpy as np
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.typing as nt
    import neuronxcc.nki.isa as nisa
    from neuronxcc.nki import trace
    from neuronxcc.nki.language import par_dim
    v5 = nl.ndarray((10944, 2048), dtype=np.float32, buffer=nl.shared_hbm)
    v6 = nl.ndarray((nl.par_dim(128), 1), dtype=np.float32, name='memset.643', buffer=nl.sbuf)
    v7 = nl.ndarray((86, 2, nl.par_dim(128), 1024), dtype=np.float32, name='theta_local_608', buffer=nl.sbuf)
    v8 = nl.ndarray((86, 2, nl.par_dim(128), 1024), dtype=np.float32, name='m_local_602', buffer=nl.sbuf)
    v9 = nl.ndarray((86, 2, nl.par_dim(128), 1024), dtype=np.float32, name='v_local_596', buffer=nl.sbuf)
    v10 = nl.ndarray((86, 2, nl.par_dim(128), 1024), dtype=np.float32, name='g_local_590', buffer=nl.sbuf)
    v11 = nl.ndarray((86, 2, nl.par_dim(128), 1024), dtype=np.float32, name='', buffer=nl.sbuf)
    v12 = nl.ndarray((86, 2, 2, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v13 = nl.ndarray((86, 2, 2, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v14 = nl.ndarray((86, 2, 2, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v15 = nl.ndarray((86, 2, 2, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v16 = nl.ndarray((86, 2, 2, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v17 = nl.ndarray((86, 2, 2, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v18 = nl.ndarray((86, 2, 2, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v19 = nl.ndarray((86, 2, 2, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v20 = nl.ndarray((86, 2, 2, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v21 = nl.ndarray((86, 2, 2, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v22 = nl.ndarray((86, 2, 2, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v23 = nl.ndarray((86, 2, 2, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v24 = nl.ndarray((86, 2, 2, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v25 = nl.ndarray((86, 2, 2, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v6[nl.arange(128)[:, None], 0] = nisa.memset(shape=(128, 1), value=np.dtype(np.uint16).type(0), dtype=np.float32, mask=None, engine=nki.isa.unknown_engine)
    for i0 in nl.affine_range(86):
        for i1 in nl.affine_range(2):
            v7[i0, i1, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nl.load(v1[128 * i0 + nl.arange(128)[:, None], 1024 * i1 + nl.arange(1024)[None, :]], dtype=np.float32, mask=-128 * i0 + -1 * nl.arange(128)[:, None] + 10943 >= 0)
            v8[i0, i1, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nl.load(v3[128 * i0 + nl.arange(128)[:, None], 1024 * i1 + nl.arange(1024)[None, :]], dtype=np.float32, mask=-128 * i0 + -1 * nl.arange(128)[:, None] + 10943 >= 0)
            v9[i0, i1, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nl.load(v4[128 * i0 + nl.arange(128)[:, None], 1024 * i1 + nl.arange(1024)[None, :]], dtype=np.float32, mask=-128 * i0 + -1 * nl.arange(128)[:, None] + 10943 >= 0)
            v10[i0, i1, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nl.load(v2[128 * i0 + nl.arange(128)[:, None], 1024 * i1 + nl.arange(1024)[None, :]], dtype=np.float32, mask=-128 * i0 + -1 * nl.arange(128)[:, None] + 10943 >= 0)
            for i2 in nl.affine_range(2):
                v12[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.tensor_scalar(data=v7[i0, i1, nl.arange(128)[:, None], 512 * i2 + nl.arange(512)[None, :]], op0=nl.multiply, operand0=np.dtype(np.float32).type(1e-05), reverse0=True, dtype=np.float32, mask=-128 * i0 + -1 * nl.arange(128)[:, None] + 10943 >= 0, engine=nki.isa.unknown_engine)
                v13[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.subtract(v7[i0, i1, nl.arange(128)[:, None], 512 * i2 + nl.arange(512)[None, :]], v12[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=-128 * i0 + -1 * nl.arange(128)[:, None] + 10943 >= 0, dtype=np.float32)
                v14[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.tensor_scalar(data=v8[i0, i1, nl.arange(128)[:, None], 512 * i2 + nl.arange(512)[None, :]], op0=nl.multiply, operand0=np.dtype(np.float32).type(0.9), reverse0=True, dtype=np.float32, mask=-128 * i0 + -1 * nl.arange(128)[:, None] + 10943 >= 0, engine=nki.isa.unknown_engine)
                v15[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.tensor_scalar(data=v10[i0, i1, nl.arange(128)[:, None], 512 * i2 + nl.arange(512)[None, :]], op0=nl.multiply, operand0=np.dtype(np.float32).type(0.1), reverse0=True, dtype=np.float32, mask=-128 * i0 + -1 * nl.arange(128)[:, None] + 10943 >= 0, engine=nki.isa.unknown_engine)
                v16[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.add(v14[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]], v15[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=-128 * i0 + -1 * nl.arange(128)[:, None] + 10943 >= 0, dtype=np.float32)
                v17[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.tensor_scalar(data=v16[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]], op0=nl.multiply, operand0=np.dtype(np.float32).type(0.01), reverse0=True, dtype=np.float32, mask=-128 * i0 + -1 * nl.arange(128)[:, None] + 10943 >= 0, engine=nki.isa.unknown_engine)
                v18[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.tensor_scalar(data=v9[i0, i1, nl.arange(128)[:, None], 512 * i2 + nl.arange(512)[None, :]], op0=nl.multiply, operand0=np.dtype(np.float32).type(0.999), reverse0=True, dtype=np.float32, mask=-128 * i0 + -1 * nl.arange(128)[:, None] + 10943 >= 0, engine=nki.isa.unknown_engine)
                v19[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.tensor_scalar(data=v10[i0, i1, nl.arange(128)[:, None], 512 * i2 + nl.arange(512)[None, :]], op0=nl.multiply, operand0=np.dtype(np.float32).type(0.001), reverse0=True, dtype=np.float32, mask=-128 * i0 + -1 * nl.arange(128)[:, None] + 10943 >= 0, engine=nki.isa.unknown_engine)
                v20[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.multiply(v19[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]], v10[i0, i1, nl.arange(128)[:, None], 512 * i2 + nl.arange(512)[None, :]], mask=-128 * i0 + -1 * nl.arange(128)[:, None] + 10943 >= 0, dtype=np.float32)
                v21[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.add(v18[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]], v20[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=-128 * i0 + -1 * nl.arange(128)[:, None] + 10943 >= 0, dtype=np.float32)
                v22[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.activation(op=nl.sqrt, data=v21[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]], bias=v6[nl.arange(128)[:, None], 0], scale=1000.0, mask=-128 * i0 + -1 * nl.arange(128)[:, None] + 10943 >= 0, dtype=np.float32)
                v23[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.tensor_scalar(data=v22[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]], op0=nl.add, operand0=np.dtype(np.float32).type(1e-08), reverse0=False, dtype=np.float32, mask=-128 * i0 + -1 * nl.arange(128)[:, None] + 10943 >= 0, engine=nki.isa.unknown_engine)
                v24[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.reciprocal(data=v23[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=-128 * i0 + -1 * nl.arange(128)[:, None] + 10943 >= 0, dtype=np.float32)
                v25[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.multiply(v17[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]], v24[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=-128 * i0 + -1 * nl.arange(128)[:, None] + 10943 >= 0, dtype=np.float32)
                v11[i0, i1, nl.arange(128)[:, None], 512 * i2 + nl.arange(512)[None, :]] = nl.subtract(v13[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]], v25[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=-128 * i0 + -1 * nl.arange(128)[:, None] + 10943 >= 0, dtype=np.float32)
                ' end loop i2 '
            nl.store(v5[128 * i0 + nl.arange(128)[:, None], 1024 * i1 + nl.arange(1024)[None, :]], value=v11[i0, i1, nl.arange(128)[:, None], nl.arange(1024)[None, :]], mask=-128 * i0 + -1 * nl.arange(128)[:, None] + 10943 >= 0)
            ' end loop i1 '
        ' end loop i0 '
    return v5