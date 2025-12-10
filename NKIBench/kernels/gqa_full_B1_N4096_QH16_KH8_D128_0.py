import numpy as np
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.typing as nt
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import trace
from neuronxcc.nki.language import par_dim

@nki.jit
def kernel(v1, v2, v3):
    import numpy as np
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.typing as nt
    import neuronxcc.nki.isa as nisa
    from neuronxcc.nki import trace
    from neuronxcc.nki.language import par_dim
    v4 = nl.ndarray((1, 8, 2, 32, 128, 128), dtype=np.float32, buffer=nl.shared_hbm)
    v5 = nl.shared_constant(np.identity(128, dtype=np.float32))
    v6 = nl.ndarray((nl.par_dim(128), 128), dtype=np.float32, name='identity_local_224', buffer=nl.sbuf)
    v7 = nl.ndarray((32, 2, nl.par_dim(128), 1024), dtype=np.float32, name='146.215', buffer=nl.sbuf)
    v8 = nl.zeros((32, 2, 8, nl.par_dim(128), 128), dtype=np.float32, name='146.220', buffer=nl.psum, lazy_initialization=True)
    v9 = nl.ndarray((2, 32, nl.par_dim(128), 8, 128), dtype=np.float32, name='q_pftranspose_146', buffer=nl.sbuf)
    v10 = nl.ndarray((2, 4, 4, nl.par_dim(128), 1024), dtype=np.float32, name='150.229', buffer=nl.sbuf)
    v11 = nl.zeros((2, 4, 4, 8, nl.par_dim(128), 128), dtype=np.float32, name='150.234', buffer=nl.psum, lazy_initialization=True)
    v12 = nl.ndarray((2, 4, nl.par_dim(128), 8, 512), dtype=np.float32, name='k_pftranspose_150', buffer=nl.sbuf)
    v13 = nl.zeros((32, 2, 4, 2, 2, 4, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.psum, lazy_initialization=True)
    v14 = nl.ndarray((32, 2, 4, 2, 2, 4, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v15 = nl.ndarray((2, 4, 2, 32, nl.par_dim(128), 1), dtype=np.float32, name='', buffer=nl.sbuf)
    v16 = nl.ndarray((8, 4, nl.par_dim(128), 1024), dtype=np.float32, name='v_local_169', buffer=nl.sbuf)
    v17 = nl.ndarray((2, 2, 4, 32, nl.par_dim(128), 1), dtype=np.float32, name='', buffer=nl.sbuf)
    v18 = nl.ndarray((32, 2, 4, 2, 2, 4, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v19 = nl.ndarray((2, 4, 2, 32, nl.par_dim(128), 1), dtype=np.float32, name='', buffer=nl.sbuf)
    v20 = nl.ndarray((2, 4, 2, 32, nl.par_dim(128), 1), dtype=np.float32, name='', buffer=nl.sbuf)
    v21 = nl.ndarray((32, 2, 4, 2, 8, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v22 = nl.zeros((2, 2, 4, 32, 2, 4, 4, nl.par_dim(128), 128), dtype=np.float32, name='135.238', buffer=nl.psum, lazy_initialization=True)
    v23 = nl.ndarray((8, 4, 2, 4, 2, 32, nl.par_dim(128), 128), dtype=np.float32, name='t51_pftranspose_135', buffer=nl.sbuf)
    v24 = nl.zeros((2, 2, 4, 32, nl.par_dim(128), 128), dtype=np.float32, name='', buffer=nl.psum, lazy_initialization=True)
    v25 = nl.ndarray((2, 4, 2, 32, nl.par_dim(128), 128), dtype=np.float32, name='', buffer=nl.sbuf)
    v26 = nl.ndarray((2, 2, 4, 4, 2, 32, nl.par_dim(128), 1), dtype=np.float32, name='', buffer=nl.sbuf)
    v27 = nl.ndarray((2, 2, 4, 32, 2, 4, nl.par_dim(128), 1), dtype=np.float32, name='', buffer=nl.sbuf)
    v6[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(v5[nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32, mask=None)
    for i0 in nl.affine_range(32):
        for i1 in nl.affine_range(2):
            v7[i0, i1, nl.arange(128)[:, None, None], 128 * nl.arange(8)[None, :, None] + nl.arange(128)[None, None, :]] = nl.load(v1[i0, nl.arange(128)[:, None, None], 8 * i1 + nl.arange(8)[None, :, None], nl.arange(128)[None, None, :]], dtype=np.float32, mask=None)
            for i2 in nl.affine_range(8):
                v8[i0, i1, i2, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(v7[i0, i1, nl.arange(128)[:, None], 128 * i2 + nl.arange(128)[None, :]], v6[nl.arange(128)[:, None], nl.arange(128)[None, :]], is_stationary_onezero=False, is_moving_onezero=True, mask=None, is_transpose=True)
                v9[i1, i0, nl.arange(128)[:, None], i2, nl.arange(128)[None, :]] = nl.copy(v8[i0, i1, i2, nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32, mask=None)
                ' end loop i2 '
            ' end loop i1 '
        ' end loop i0 '
    for i3 in nl.affine_range(2):
        for i4 in nl.affine_range(4):
            for i5 in nl.affine_range(4):
                v10[i3, i4, i5, nl.arange(128)[:, None, None], 128 * nl.arange(8)[None, :, None] + nl.arange(128)[None, None, :]] = nl.load(v2[0, i4 + 4 * i3, i5, nl.arange(128)[:, None, None], nl.arange(8)[None, :, None], nl.arange(128)[None, None, :]], dtype=np.float32, mask=None)
                for i6 in nl.affine_range(8):
                    v11[i3, i4, i5, i6, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(v10[i3, i4, i5, nl.arange(128)[:, None], 128 * i6 + nl.arange(128)[None, :]], v6[nl.arange(128)[:, None], nl.arange(128)[None, :]], is_stationary_onezero=False, is_moving_onezero=True, mask=None, is_transpose=True)
                    v12[i3, i4, nl.arange(128)[:, None], i6, 128 * i5 + nl.arange(128)[None, :]] = nl.copy(v11[i3, i4, i5, i6, nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32, mask=None)
                    ' end loop i6 '
                ' end loop i5 '
            ' end loop i4 '
        for i7 in nl.affine_range(2):
            for i8 in nl.affine_range(4):
                for i9 in nl.affine_range(4):
                    for i10 in nl.affine_range(2):
                        for i11 in nl.affine_range(32):
                            v13[i11, i7, i8, i10, i3, i9, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.nc_matmul(v9[i7, i11, nl.arange(128)[:, None], i10 + 2 * i8, nl.arange(128)[None, :]], v12[i3, i9, nl.arange(128)[:, None], i8 + 4 * i7, nl.arange(512)[None, :]], is_stationary_onezero=False, is_moving_onezero=False, mask=None)
                            v14[i11, i7, i8, i10, i3, i9, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.tensor_scalar(data=v13[i11, i7, i8, i10, i3, i9, nl.arange(128)[:, None], nl.arange(512)[None, :]], op0=nl.multiply, operand0=np.dtype(np.float32).type(0.08838835154663706), reverse0=False, dtype=np.float32, mask=None, engine=nki.isa.unknown_engine)
                            v26[i3, i7, i8, i9, i10, i11, nl.arange(128)[:, None], 0] = nisa.tensor_reduce(nl.max, data=v14[i11, i7, i8, i10, i3, i9, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=None, axis=[1], dtype=np.float32, negate=False)
                            v15[i7, i8, i10, i11, nl.arange(128)[:, None], 0] = nl.loop_reduce(v26[i3, i7, i8, i9, i10, i11, nl.arange(128)[:, None], 0], op=np.max, loop_indices=[i3, i9], mask=None, dtype=np.float32)
                            ' end loop i11 '
                        ' end loop i10 '
                    ' end loop i9 '
                ' end loop i8 '
            ' end loop i7 '
        ' end loop i3 '
    for i12 in nl.affine_range(8):
        for i13 in nl.affine_range(4):
            v16[i12, i13, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nl.load(v3[0, 4 * i12 + i13, nl.arange(128)[:, None], nl.arange(1024)[None, :]], dtype=np.float32, mask=None)
            ' end loop i13 '
        ' end loop i12 '
    for i14 in nl.affine_range(2):
        for i15 in nl.affine_range(2):
            for i16 in nl.affine_range(4):
                for i17 in nl.affine_range(32):
                    v17[i14, i15, i16, i17, nl.arange(128)[:, None], 0] = nisa.tensor_scalar(data=v15[i15, i16, i14, i17, nl.arange(128)[:, None], 0], op0=nl.maximum, operand0=np.dtype(np.float32).type(-3.4028235e+38), reverse0=False, op1=nl.multiply, operand1=np.dtype(np.float32).type(-1.0), reverse1=False, dtype=np.float32, mask=None, engine=nki.isa.unknown_engine)
                    for i18 in nl.affine_range(2):
                        for i19 in nl.affine_range(4):
                            v18[i17, i15, i16, i14, i18, i19, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.activation(op=nl.exp, data=v14[i17, i15, i16, i14, i18, i19, nl.arange(128)[:, None], nl.arange(512)[None, :]], bias=v17[i14, i15, i16, i17, nl.arange(128)[:, None], 0], scale=1.0, mask=None, dtype=np.float32)
                            v27[i14, i15, i16, i17, i18, i19, nl.arange(128)[:, None], 0] = nisa.tensor_reduce(nl.add, data=v18[i17, i15, i16, i14, i18, i19, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=None, axis=[1], dtype=np.float32, negate=False)
                            v19[i15, i16, i14, i17, nl.arange(128)[:, None], 0] = nl.loop_reduce(v27[i14, i15, i16, i17, i18, i19, nl.arange(128)[:, None], 0], op=np.add, loop_indices=[i19, i18], mask=None, dtype=np.float32)
                            ' end loop i19 '
                        ' end loop i18 '
                    v20[i15, i16, i14, i17, nl.arange(128)[:, None], 0] = nisa.reciprocal(data=v19[i15, i16, i14, i17, nl.arange(128)[:, None], 0], mask=None, dtype=np.float32)
                    for i20 in nl.affine_range(2):
                        for i21 in nl.affine_range(4):
                            v21[i17, i15, i16, i14, 4 * i20 + i21, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.tensor_scalar(data=v18[i17, i15, i16, i14, i20, i21, nl.arange(128)[:, None], nl.arange(512)[None, :]], op0=nl.multiply, operand0=v20[i15, i16, i14, i17, nl.arange(128)[:, None], 0], reverse0=False, dtype=np.float32, mask=None, engine=nki.isa.unknown_engine)
                            for i22 in nl.affine_range(4):
                                v22[i14, i15, i16, i17, i20, i21, i22, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(v21[i17, i15, i16, i14, 4 * i20 + i21, nl.arange(128)[:, None], 128 * i22 + nl.arange(128)[None, :]], v6[nl.arange(128)[:, None], nl.arange(128)[None, :]], is_stationary_onezero=False, is_moving_onezero=True, mask=None, is_transpose=True)
                                v23[4 * i20 + i21, i22, i15, i16, i14, i17, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.copy(v22[i14, i15, i16, i17, i20, i21, i22, nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32, mask=None)
                                ' end loop i22 '
                            ' end loop i21 '
                        ' end loop i20 '
                    for i23 in nl.affine_range(8):
                        for i24 in nl.affine_range(4):
                            v24[i14, i15, i16, i17, nl.arange(128)[:, None], nl.arange(128)[None, :]] += nisa.nc_matmul(v23[i23, i24, i15, i16, i14, i17, nl.arange(128)[:, None], nl.arange(128)[None, :]], v16[i23, i24, nl.arange(128)[:, None], 128 * i16 + 512 * i15 + nl.arange(128)[None, :]], is_stationary_onezero=False, is_moving_onezero=False, mask=None)
                            ' end loop i24 '
                        ' end loop i23 '
                    v25[i15, i16, i14, i17, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.copy(v24[i14, i15, i16, i17, nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32, mask=None)
                    nl.store(v4[0, i16 + 4 * i15, i14, i17, nl.arange(128)[:, None], nl.arange(128)[None, :]], value=v25[i15, i16, i14, i17, nl.arange(128)[:, None], nl.arange(128)[None, :]], mask=None)
                    ' end loop i17 '
                ' end loop i16 '
            ' end loop i15 '
        ' end loop i14 '
    return v4