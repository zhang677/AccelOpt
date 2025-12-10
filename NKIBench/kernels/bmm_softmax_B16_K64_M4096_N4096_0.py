import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim

@nki.jit
def kernel(v1, v2):
    import numpy as np
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.typing as nt
    import neuronxcc.nki.isa as nisa
    from neuronxcc.nki import trace
    from neuronxcc.nki.language import par_dim
    v3 = nl.ndarray((16, 32, 128, 4096), dtype=np.float32, buffer=nl.shared_hbm)
    v4 = nl.shared_constant(np.identity(128, dtype=np.float32))
    v5 = nl.ndarray((nl.par_dim(128), 128), dtype=np.float32, name='identity_local_140', buffer=nl.sbuf)
    v6 = nl.ndarray((16, 4, nl.par_dim(64), 1024), dtype=np.float32, name='rhs_local_89', buffer=nl.sbuf)
    v7 = nl.ndarray((4, 16, 4, 8, nl.par_dim(128), 64), dtype=np.float32, name='', buffer=nl.sbuf)
    v8 = nl.zeros((16, 4, 4, 8, nl.par_dim(64), 128), dtype=np.float32, name='73.136', buffer=nl.psum, lazy_initialization=True)
    v9 = nl.ndarray((4, 16, 4, nl.par_dim(64), 1024), dtype=np.float32, name='', buffer=nl.sbuf)
    v10 = nl.zeros((16, 4, 4, 8, 2, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.psum, lazy_initialization=True)
    v11 = nl.ndarray((4, 8, 16, 4, 2, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v12 = nl.ndarray((16, 4, 8, nl.par_dim(128), 1), dtype=np.float32, name='', buffer=nl.sbuf)
    v13 = nl.ndarray((16, 4, 8, nl.par_dim(128), 1), dtype=np.float32, name='', buffer=nl.sbuf)
    v14 = nl.ndarray((4, 8, 16, 4, 2, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v15 = nl.ndarray((16, 4, 8, nl.par_dim(128), 1), dtype=np.float32, name='', buffer=nl.sbuf)
    v16 = nl.ndarray((16, 4, 8, nl.par_dim(128), 1), dtype=np.float32, name='', buffer=nl.sbuf)
    v17 = nl.ndarray((16, 4, 8, 4, nl.par_dim(128), 1024), dtype=np.float32, name='', buffer=nl.sbuf)
    v18 = nl.ndarray((16, 4, 4, 8, 2, nl.par_dim(128), 1), dtype=np.float32, name='', buffer=nl.sbuf)
    v19 = nl.ndarray((16, 4, 8, 4, 2, nl.par_dim(128), 1), dtype=np.float32, name='', buffer=nl.sbuf)
    v5[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(v4[nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32, mask=None)
    for i0 in nl.affine_range(16):
        for i1 in nl.affine_range(4):
            v6[i0, i1, nl.arange(64)[:, None], nl.arange(1024)[None, :]] = nl.load(v2[i0, nl.arange(64)[:, None], 1024 * i1 + nl.arange(1024)[None, :]], dtype=np.float32, mask=None)
            for i2 in nl.affine_range(4):
                for i3 in nl.affine_range(8):
                    v7[i1, i0, i2, i3, nl.arange(128)[:, None], nl.arange(64)[None, :]] = nl.load(v1[i0, 1024 * i2 + 128 * i3 + nl.arange(128)[:, None], nl.arange(64)[None, :]], dtype=np.float32, mask=None)
                    v8[i0, i1, i2, i3, nl.arange(64)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(v7[i1, i0, i2, i3, nl.arange(128)[:, None], nl.arange(64)[None, :]], v5[nl.arange(128)[:, None], nl.arange(128)[None, :]], is_stationary_onezero=False, is_moving_onezero=True, mask=None, is_transpose=True)
                    v9[i1, i0, i2, nl.arange(64)[:, None], 128 * i3 + nl.arange(128)[None, :]] = nl.copy(v8[i0, i1, i2, i3, nl.arange(64)[:, None], nl.arange(128)[None, :]], dtype=np.float32, mask=None)
                    ' end loop i3 '
                for i4 in nl.affine_range(8):
                    for i5 in nl.affine_range(2):
                        v10[i0, i1, i2, i4, i5, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.nc_matmul(v9[i1, i0, i2, nl.arange(64)[:, None], 128 * i4 + nl.arange(128)[None, :]], v6[i0, i1, nl.arange(64)[:, None], 512 * i5 + nl.arange(512)[None, :]], is_stationary_onezero=False, is_moving_onezero=False, mask=None)
                        v11[i2, i4, i0, i1, i5, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.copy(v10[i0, i1, i2, i4, i5, nl.arange(128)[:, None], nl.arange(512)[None, :]], dtype=np.float32, mask=None)
                        v18[i0, i1, i2, i4, i5, nl.arange(128)[:, None], 0] = nisa.tensor_reduce(nl.max, data=v11[i2, i4, i0, i1, i5, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=None, axis=[1], dtype=np.float32, negate=False)
                        v12[i0, i2, i4, nl.arange(128)[:, None], 0] = nl.loop_reduce(v18[i0, i1, i2, i4, i5, nl.arange(128)[:, None], 0], op=np.max, loop_indices=[i1, i5], mask=None, dtype=np.float32)
                        ' end loop i5 '
                    ' end loop i4 '
                ' end loop i2 '
            ' end loop i1 '
        for i6 in nl.affine_range(4):
            for i7 in nl.affine_range(8):
                v13[i0, i6, i7, nl.arange(128)[:, None], 0] = nisa.tensor_scalar(data=v12[i0, i6, i7, nl.arange(128)[:, None], 0], op0=nl.maximum, operand0=np.dtype(np.float32).type(-3.4028235e+38), reverse0=False, op1=nl.multiply, operand1=np.dtype(np.float32).type(-1.0), reverse1=False, dtype=np.float32, mask=None, engine=nki.isa.unknown_engine)
                for i8 in nl.affine_range(4):
                    for i9 in nl.affine_range(2):
                        v14[i6, i7, i0, i8, i9, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.activation(op=nl.exp, data=v11[i6, i7, i0, i8, i9, nl.arange(128)[:, None], nl.arange(512)[None, :]], bias=v13[i0, i6, i7, nl.arange(128)[:, None], 0], scale=1.0, mask=None, dtype=np.float32)
                        v19[i0, i6, i7, i8, i9, nl.arange(128)[:, None], 0] = nisa.tensor_reduce(nl.add, data=v14[i6, i7, i0, i8, i9, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=None, axis=[1], dtype=np.float32, negate=False)
                        v15[i0, i6, i7, nl.arange(128)[:, None], 0] = nl.loop_reduce(v19[i0, i6, i7, i8, i9, nl.arange(128)[:, None], 0], op=np.add, loop_indices=[i8, i9], mask=None, dtype=np.float32)
                        ' end loop i9 '
                    ' end loop i8 '
                v16[i0, i6, i7, nl.arange(128)[:, None], 0] = nisa.reciprocal(data=v15[i0, i6, i7, nl.arange(128)[:, None], 0], mask=None, dtype=np.float32)
                for i10 in nl.affine_range(4):
                    for i11 in nl.affine_range(2):
                        v17[i0, i6, i7, i10, nl.arange(128)[:, None], 512 * i11 + nl.arange(512)[None, :]] = nisa.tensor_scalar(data=v14[i6, i7, i0, i10, i11, nl.arange(128)[:, None], nl.arange(512)[None, :]], op0=nl.multiply, operand0=v16[i0, i6, i7, nl.arange(128)[:, None], 0], reverse0=False, dtype=np.float32, mask=None, engine=nki.isa.unknown_engine)
                        ' end loop i11 '
                    nl.store(v3[i0, 8 * i6 + i7, nl.arange(128)[:, None], 1024 * i10 + nl.arange(1024)[None, :]], value=v17[i0, i6, i7, i10, nl.arange(128)[:, None], nl.arange(1024)[None, :]], mask=None)
                    ' end loop i10 '
                ' end loop i7 '
            ' end loop i6 '
        ' end loop i0 '
    return v3