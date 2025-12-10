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
    v3 = nl.ndarray((32, 128, 2048), dtype=np.float32, buffer=nl.shared_hbm)
    v4 = nl.ndarray((nl.par_dim(128), 1), dtype=np.float32, name='memset.174', buffer=nl.sbuf)
    v5 = nl.shared_constant(np.identity(128, dtype=np.float32))
    v6 = nl.ndarray((nl.par_dim(128), 128), dtype=np.float32, name='identity_local_168', buffer=nl.sbuf)
    v7 = nl.ndarray((4, 8, nl.par_dim(128), 1024), dtype=np.float32, name='input_tensor_local_112', buffer=nl.sbuf)
    v8 = nl.ndarray((4, 8, nl.par_dim(128), 1), dtype=np.float32, name='', buffer=nl.sbuf)
    v9 = nl.ndarray((4, 8, nl.par_dim(128), 1), dtype=np.float32, name='', buffer=nl.sbuf)
    v10 = nl.ndarray((4, 8, nl.par_dim(128), 1024), dtype=np.float32, name='input_tensor_local_118', buffer=nl.sbuf)
    v11 = nl.ndarray((4, 8, 2, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v12 = nl.zeros((4, 8, 2, 4, nl.par_dim(128), 128), dtype=np.float32, name='93.164', buffer=nl.psum, lazy_initialization=True)
    v13 = nl.ndarray((2, 4, 4, 8, nl.par_dim(128), 128), dtype=np.float32, name='t26_pftranspose_93', buffer=nl.sbuf)
    v14 = nl.ndarray((4, 2, 2, 4, nl.par_dim(128), 1024), dtype=np.float32, name='', buffer=nl.sbuf)
    v15 = nl.zeros((2, 4, 8, 2, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.psum, lazy_initialization=True)
    v16 = nl.ndarray((4, 8, 2, nl.par_dim(128), 1024), dtype=np.float32, name='', buffer=nl.sbuf)
    v17 = nl.ndarray((4, 8, 2, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v18 = nl.ndarray((4, 8, 2, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v19 = nl.ndarray((4, 8, 2, nl.par_dim(128), 1), dtype=np.float32, name='', buffer=nl.sbuf)
    v4[nl.arange(128)[:, None], 0] = nisa.memset(shape=(128, 1), value=np.dtype(np.uint16).type(0), dtype=np.float32, mask=None, engine=nki.isa.unknown_engine)
    v6[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(v5[nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32, mask=None)
    for i0 in nl.affine_range(4):
        for i1 in nl.affine_range(8):
            v7[i0, i1, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nl.load(v1[8 * i0 + i1, nl.arange(128)[:, None], nl.arange(1024)[None, :]], dtype=np.float32, mask=None)
            for i2 in nl.affine_range(2):
                v17[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.activation(op=nl.square, data=v7[i0, i1, nl.arange(128)[:, None], 512 * i2 + nl.arange(512)[None, :]], bias=v4[nl.arange(128)[:, None], 0], scale=1.0, mask=None, dtype=np.float32)
                v18[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.tensor_scalar(data=v17[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]], op0=nl.multiply, operand0=np.dtype(np.float32).type(0.0009765625), reverse0=False, dtype=np.float32, mask=None, engine=nki.isa.unknown_engine)
                v19[i0, i1, i2, nl.arange(128)[:, None], 0] = nisa.tensor_reduce(nl.add, data=v18[i0, i1, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=None, axis=[1], dtype=np.float32, negate=False)
                v8[i0, i1, nl.arange(128)[:, None], 0] = nl.loop_reduce(v19[i0, i1, i2, nl.arange(128)[:, None], 0], op=np.add, loop_indices=[i2], mask=None, dtype=np.float32)
                ' end loop i2 '
            v9[i0, i1, nl.arange(128)[:, None], 0] = nisa.activation(op=nl.rsqrt, data=v8[i0, i1, nl.arange(128)[:, None], 0], bias=v4[nl.arange(128)[:, None], 0], scale=1.0, mask=None, dtype=np.float32)
            v10[i0, i1, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nl.load(v1[8 * i0 + i1, nl.arange(128)[:, None], nl.arange(1024)[None, :]], dtype=np.float32, mask=None)
            for i3 in nl.affine_range(2):
                v11[i0, i1, i3, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.tensor_scalar(data=v10[i0, i1, nl.arange(128)[:, None], 512 * i3 + nl.arange(512)[None, :]], op0=nl.multiply, operand0=v9[i0, i1, nl.arange(128)[:, None], 0], reverse0=False, dtype=np.float32, mask=None, engine=nki.isa.unknown_engine)
                for i4 in nl.affine_range(4):
                    v12[i0, i1, i3, i4, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(v11[i0, i1, i3, nl.arange(128)[:, None], 128 * i4 + nl.arange(128)[None, :]], v6[nl.arange(128)[:, None], nl.arange(128)[None, :]], is_stationary_onezero=False, is_moving_onezero=True, mask=None, is_transpose=True)
                    v13[i3, i4, i0, i1, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.copy(v12[i0, i1, i3, i4, nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32, mask=None)
                    ' end loop i4 '
                ' end loop i3 '
            ' end loop i1 '
        for i5 in nl.affine_range(2):
            for i6 in nl.affine_range(2):
                for i7 in nl.affine_range(4):
                    v14[i0, i5, i6, i7, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nl.load(v2[i7 + 4 * i6, nl.arange(128)[:, None], 1024 * i5 + nl.arange(1024)[None, :]], dtype=np.float32, mask=None)
                    ' end loop i7 '
                ' end loop i6 '
            for i8 in nl.affine_range(8):
                for i9 in nl.affine_range(2):
                    for i10 in nl.affine_range(2):
                        for i11 in nl.affine_range(4):
                            v15[i5, i0, i8, i9, nl.arange(128)[:, None], nl.arange(512)[None, :]] += nisa.nc_matmul(v13[i10, i11, i0, i8, nl.arange(128)[:, None], nl.arange(128)[None, :]], v14[i0, i5, i10, i11, nl.arange(128)[:, None], 512 * i9 + nl.arange(512)[None, :]], is_stationary_onezero=False, is_moving_onezero=False, mask=None)
                            ' end loop i11 '
                        ' end loop i10 '
                    v16[i0, i8, i5, nl.arange(128)[:, None], 512 * i9 + nl.arange(512)[None, :]] = nl.copy(v15[i5, i0, i8, i9, nl.arange(128)[:, None], nl.arange(512)[None, :]], dtype=np.float32, mask=None)
                    ' end loop i9 '
                nl.store(v3[i8 + 8 * i0, nl.arange(128)[:, None], 1024 * i5 + nl.arange(1024)[None, :]], value=v16[i0, i8, i5, nl.arange(128)[:, None], nl.arange(1024)[None, :]], mask=None)
                ' end loop i8 '
            ' end loop i5 '
        ' end loop i0 '
    return v3