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
    v5 = nl.ndarray((32, 128, 1024), dtype=np.float32, buffer=nl.shared_hbm)
    v6 = nl.ndarray((nl.par_dim(128), 1), dtype=np.float32, name='memset.200', buffer=nl.sbuf)
    v7 = nl.shared_constant(np.identity(128, dtype=np.float32))
    v8 = nl.ndarray((nl.par_dim(128), 128), dtype=np.float32, name='identity_local_182', buffer=nl.sbuf)
    v9 = nl.ndarray((3, 8, nl.par_dim(128), 1024), dtype=np.float32, name='w_gate_local_109', buffer=nl.sbuf)
    v10 = nl.ndarray((3, 8, 4, nl.par_dim(128), 1024), dtype=np.float32, name='', buffer=nl.sbuf)
    v11 = nl.zeros((3, 8, 4, 8, nl.par_dim(128), 128), dtype=np.float32, name='95.178', buffer=nl.psum, lazy_initialization=True)
    v12 = nl.ndarray((3, 8, nl.par_dim(128), 8, 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v13 = nl.zeros((3, 8, 8, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.psum, lazy_initialization=True)
    v14 = nl.ndarray((3, 8, 8, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v15 = nl.ndarray((3, 8, nl.par_dim(128), 1024), dtype=np.float32, name='w_up_local_118', buffer=nl.sbuf)
    v16 = nl.ndarray((3, 8, 4, nl.par_dim(128), 1024), dtype=np.float32, name='', buffer=nl.sbuf)
    v17 = nl.zeros((3, 8, 4, 8, nl.par_dim(128), 128), dtype=np.float32, name='95.193', buffer=nl.psum, lazy_initialization=True)
    v18 = nl.ndarray((3, 8, nl.par_dim(128), 8, 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v19 = nl.zeros((3, 8, 8, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.psum, lazy_initialization=True)
    v20 = nl.ndarray(shape=(3, 8, 8, 128, 512), dtype=np.float32, name='_spill_163', buffer=nl.hbm)
    v21 = nl.ndarray((3, 8, nl.par_dim(128), 1024), dtype=np.float32, name='w_down_local_126', buffer=nl.sbuf)
    v22 = nl.ndarray((3, 8, 8, nl.par_dim(128), 512), dtype=np.float32, name='_reload_166', buffer=nl.sbuf)
    v23 = nl.zeros((8, 4, 2, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.psum, lazy_initialization=True)
    v24 = nl.ndarray((8, 4, nl.par_dim(128), 1024), dtype=np.float32, name='', buffer=nl.sbuf)
    v25 = nl.ndarray((3, 8, 8, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v26 = nl.ndarray((3, 8, 8, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v27 = nl.ndarray((3, 8, 8, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v28 = nl.ndarray((3, 8, 8, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v29 = nl.ndarray((3, 8, 8, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v6[nl.arange(128)[:, None], 0] = nisa.memset(shape=(128, 1), value=np.dtype(np.uint16).type(0), dtype=np.float32, mask=None, engine=nki.isa.unknown_engine)
    v8[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(v7[nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32, mask=None)
    for i0 in nl.affine_range(3):
        for i1 in nl.affine_range(8):
            v9[i0, i1, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nl.load(v4[i1, nl.arange(128)[:, None], 1024 * i0 + nl.arange(1024)[None, :]], dtype=np.float32, mask=None)
            ' end loop i1 '
        for i2 in nl.affine_range(8):
            for i3 in nl.affine_range(4):
                v10[i0, i2, i3, nl.arange(128)[:, None, None], 128 * nl.arange(8)[None, :, None] + nl.arange(128)[None, None, :]] = nl.load(v1[i2, i3, nl.arange(128)[:, None, None], nl.arange(8)[None, :, None], nl.arange(128)[None, None, :]], dtype=np.float32, mask=None)
                for i4 in nl.affine_range(8):
                    v11[i0, i2, i3, i4, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(v10[i0, i2, i3, nl.arange(128)[:, None], 128 * i4 + nl.arange(128)[None, :]], v8[nl.arange(128)[:, None], nl.arange(128)[None, :]], is_stationary_onezero=False, is_moving_onezero=True, mask=None, is_transpose=True)
                    v12[i0, i2, nl.arange(128)[:, None], i4, 128 * i3 + nl.arange(128)[None, :]] = nl.copy(v11[i0, i2, i3, i4, nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32, mask=None)
                    ' end loop i4 '
                ' end loop i3 '
            for i5 in nl.affine_range(8):
                for i6 in nl.affine_range(8):
                    v13[i0, i2, i5, nl.arange(128)[:, None], nl.arange(512)[None, :]] += nisa.nc_matmul(v9[i0, i6, nl.arange(128)[:, None], 128 * i5 + nl.arange(128)[None, :]], v12[i0, i2, nl.arange(128)[:, None], i6, nl.arange(512)[None, :]], is_stationary_onezero=False, is_moving_onezero=False, mask=None)
                    ' end loop i6 '
                v14[i0, i5, i2, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.copy(v13[i0, i2, i5, nl.arange(128)[:, None], nl.arange(512)[None, :]], dtype=np.float32, mask=None)
                ' end loop i5 '
            ' end loop i2 '
        ' end loop i0 '
    for i7 in nl.affine_range(3):
        for i8 in nl.affine_range(8):
            v15[i7, i8, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nl.load(v2[i8, nl.arange(128)[:, None], 1024 * i7 + nl.arange(1024)[None, :]], dtype=np.float32, mask=None)
            ' end loop i8 '
        for i9 in nl.affine_range(8):
            for i10 in nl.affine_range(4):
                v16[i7, i9, i10, nl.arange(128)[:, None, None], 128 * nl.arange(8)[None, :, None] + nl.arange(128)[None, None, :]] = nl.load(v1[i9, i10, nl.arange(128)[:, None, None], nl.arange(8)[None, :, None], nl.arange(128)[None, None, :]], dtype=np.float32, mask=None)
                for i11 in nl.affine_range(8):
                    v17[i7, i9, i10, i11, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(v16[i7, i9, i10, nl.arange(128)[:, None], 128 * i11 + nl.arange(128)[None, :]], v8[nl.arange(128)[:, None], nl.arange(128)[None, :]], is_stationary_onezero=False, is_moving_onezero=True, mask=None, is_transpose=True)
                    v18[i7, i9, nl.arange(128)[:, None], i11, 128 * i10 + nl.arange(128)[None, :]] = nl.copy(v17[i7, i9, i10, i11, nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32, mask=None)
                    ' end loop i11 '
                ' end loop i10 '
            for i12 in nl.affine_range(8):
                for i13 in nl.affine_range(8):
                    v19[i7, i12, i9, nl.arange(128)[:, None], nl.arange(512)[None, :]] += nisa.nc_matmul(v15[i7, i13, nl.arange(128)[:, None], 128 * i12 + nl.arange(128)[None, :]], v18[i7, i9, nl.arange(128)[:, None], i13, nl.arange(512)[None, :]], is_stationary_onezero=False, is_moving_onezero=False, mask=None)
                    ' end loop i13 '
                v25[i7, i9, i12, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.activation(op=nl.exp, data=v14[i7, i12, i9, nl.arange(128)[:, None], nl.arange(512)[None, :]], bias=v6[nl.arange(128)[:, None], 0], scale=-1.0, mask=None, dtype=np.float32)
                v26[i7, i9, i12, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.tensor_scalar(data=v25[i7, i9, i12, nl.arange(128)[:, None], nl.arange(512)[None, :]], op0=nl.add, operand0=np.dtype(np.float32).type(1), reverse0=True, dtype=np.float32, mask=None, engine=nki.isa.unknown_engine)
                v27[i7, i9, i12, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.reciprocal(data=v26[i7, i9, i12, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=None, dtype=np.float32)
                v28[i7, i9, i12, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.multiply(v14[i7, i12, i9, nl.arange(128)[:, None], nl.arange(512)[None, :]], v27[i7, i9, i12, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=None, dtype=np.float32)
                v29[i7, i9, i12, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.multiply(v28[i7, i9, i12, nl.arange(128)[:, None], nl.arange(512)[None, :]], v19[i7, i12, i9, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=None, dtype=np.float32)
                nl.store(v20[i7, i12, i9, nl.arange(128)[:, None], nl.arange(512)[None, :]], value=v29[i7, i9, i12, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=None)
                ' end loop i12 '
            ' end loop i9 '
        ' end loop i7 '
    for i14 in nl.affine_range(3):
        for i15 in nl.affine_range(8):
            v21[i14, i15, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nl.load(v3[i15 + 8 * i14, nl.arange(128)[:, None], nl.arange(1024)[None, :]], dtype=np.float32, mask=None)
            ' end loop i15 '
        ' end loop i14 '
    for i16 in nl.affine_range(8):
        for i17 in nl.affine_range(3):
            for i18 in nl.affine_range(8):
                v22[i17, i18, i16, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.load(v20[i17, i18, i16, nl.arange(128)[:, None], nl.arange(512)[None, :]], dtype=np.float32, mask=None)
                ' end loop i18 '
            ' end loop i17 '
        for i19 in nl.affine_range(4):
            for i20 in nl.affine_range(2):
                for i21 in nl.affine_range(3):
                    for i22 in nl.affine_range(8):
                        v23[i16, i19, i20, nl.arange(128)[:, None], nl.arange(512)[None, :]] += nisa.nc_matmul(v22[i21, i22, i16, nl.arange(128)[:, None], 128 * i19 + nl.arange(128)[None, :]], v21[i21, i22, nl.arange(128)[:, None], 512 * i20 + nl.arange(512)[None, :]], is_stationary_onezero=False, is_moving_onezero=False, mask=None)
                        ' end loop i22 '
                    ' end loop i21 '
                v24[i16, i19, nl.arange(128)[:, None], 512 * i20 + nl.arange(512)[None, :]] = nl.copy(v23[i16, i19, i20, nl.arange(128)[:, None], nl.arange(512)[None, :]], dtype=np.float32, mask=None)
                ' end loop i20 '
            nl.store(v5[i19 + 4 * i16, nl.arange(128)[:, None], nl.arange(1024)[None, :]], value=v24[i16, i19, nl.arange(128)[:, None], nl.arange(1024)[None, :]], mask=None)
            ' end loop i19 '
        ' end loop i16 '
    return v5