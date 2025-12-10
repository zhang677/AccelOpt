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
    v5 = nl.ndarray((8, 4, 128, 96, 128), dtype=np.float32, buffer=nl.shared_hbm)
    v6 = nl.shared_constant(np.identity(128, dtype=np.float32))
    v7 = nl.ndarray((nl.par_dim(128), 128), dtype=np.float32, name='identity_local_111', buffer=nl.sbuf)
    v8 = nl.ndarray((5, 8, nl.par_dim(128), 128), dtype=np.float32, name='a_local_56', buffer=nl.sbuf)
    v9 = nl.ndarray((2, 4, 5, 4, nl.par_dim(128), 1024), dtype=np.float32, name='38.102', buffer=nl.sbuf)
    v10 = nl.zeros((2, 4, 5, 4, 8, nl.par_dim(128), 128), dtype=np.float32, name='38.107', buffer=nl.psum, lazy_initialization=True)
    v11 = nl.ndarray((2, 4, 5, nl.par_dim(128), 8, 512), dtype=np.float32, name='x_pftranspose_38', buffer=nl.sbuf)
    v12 = nl.zeros((2, 4, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.psum, lazy_initialization=True)
    v13 = nl.ndarray((2, 4, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v14 = nl.ndarray((2, 12, nl.par_dim(128), 1024), dtype=np.float32, name='', buffer=nl.sbuf)
    v15 = nl.zeros((2, 12, 4, 2, 4, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.psum, lazy_initialization=True)
    v16 = nl.ndarray((12, 2, 4, 2, 4, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v17 = nl.ndarray((12, 4, 2, 5, nl.par_dim(128), 1024), dtype=np.float32, name='w_local_73', buffer=nl.sbuf)
    v18 = nl.ndarray((12, 4, 2, 4, 2, 4, nl.par_dim(128), 640), dtype=np.float32, name='', buffer=nl.sbuf)
    v19 = nl.zeros((12, 4, 2, 4, 2, 4, 5, nl.par_dim(128), 128), dtype=np.float32, name='38.122', buffer=nl.psum, lazy_initialization=True)
    v20 = nl.ndarray((12, 2, 4, 4, 2, nl.par_dim(128), 5, 512), dtype=np.float32, name='', buffer=nl.sbuf)
    v21 = nl.zeros((12, 4, 2, 4, 2, 4, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.psum, lazy_initialization=True)
    v22 = nl.full((12, 2, 4, 2, 4, nl.par_dim(128), 512), fill_value=np.dtype(np.float32).type(0), dtype=np.float32, name='', buffer=nl.sbuf)
    v23 = nl.ndarray((2, 4, 12, 2, 4, nl.par_dim(128), 512), dtype=np.float32, name='.o0_pftranspose_42', buffer=nl.sbuf)
    v24 = nl.zeros((12, 2, 4, 2, 4, 4, nl.par_dim(128), 128), dtype=np.float32, name='42.130', buffer=nl.psum, lazy_initialization=True)
    v25 = nl.ndarray((12, 2, 4, 2, 4, 4, nl.par_dim(128), 128), dtype=np.float32, name='42.143', buffer=nl.sbuf)
    v7[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(v6[nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32, mask=None)
    for i0 in nl.affine_range(5):
        for i1 in nl.affine_range(8):
            v8[i0, i1, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(v3[8 * i0 + i1, nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32, mask=None)
            ' end loop i1 '
        ' end loop i0 '
    for i2 in nl.affine_range(2):
        for i3 in nl.affine_range(4):
            for i4 in nl.affine_range(5):
                for i5 in nl.affine_range(4):
                    v9[i2, i3, i4, i5, nl.arange(128)[:, None, None], 128 * nl.arange(8)[None, :, None] + nl.arange(128)[None, None, :]] = nl.load(v1[i3 + 4 * i2, i5, nl.arange(128)[:, None, None], 8 * i4 + nl.arange(8)[None, :, None], nl.arange(128)[None, None, :]], dtype=np.float32, mask=None)
                    for i6 in nl.affine_range(8):
                        v10[i2, i3, i4, i5, i6, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(v9[i2, i3, i4, i5, nl.arange(128)[:, None], 128 * i6 + nl.arange(128)[None, :]], v7[nl.arange(128)[:, None], nl.arange(128)[None, :]], is_stationary_onezero=False, is_moving_onezero=True, mask=None, is_transpose=True)
                        v11[i2, i3, i4, nl.arange(128)[:, None], i6, 128 * i5 + nl.arange(128)[None, :]] = nl.copy(v10[i2, i3, i4, i5, i6, nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32, mask=None)
                        ' end loop i6 '
                    ' end loop i5 '
                ' end loop i4 '
            for i7 in nl.affine_range(5):
                for i8 in nl.affine_range(8):
                    v12[i2, i3, nl.arange(128)[:, None], nl.arange(512)[None, :]] += nisa.nc_matmul(v8[i7, i8, nl.arange(128)[:, None], nl.arange(128)[None, :]], v11[i2, i3, i7, nl.arange(128)[:, None], i8, nl.arange(512)[None, :]], is_stationary_onezero=False, is_moving_onezero=False, mask=None)
                    ' end loop i8 '
                ' end loop i7 '
            v13[i2, i3, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.copy(v12[i2, i3, nl.arange(128)[:, None], nl.arange(512)[None, :]], dtype=np.float32, mask=None)
            ' end loop i3 '
        for i9 in nl.affine_range(12):
            v14[i2, i9, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nl.load(v4[nl.arange(128)[:, None], 1024 * i9 + nl.arange(1024)[None, :]], dtype=np.float32, mask=None)
            for i10 in nl.affine_range(4):
                for i11 in nl.affine_range(2):
                    for i12 in nl.affine_range(4):
                        v15[i2, i9, i10, i11, i12, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nisa.nc_matmul(v14[i2, i9, nl.arange(128)[:, None], 512 * i11 + 128 * i12 + nl.arange(128)[None, :]], v13[i2, i10, nl.arange(128)[:, None], nl.arange(512)[None, :]], is_stationary_onezero=False, is_moving_onezero=False, mask=None)
                        v16[i9, i11, i12, i2, i10, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.copy(v15[i2, i9, i10, i11, i12, nl.arange(128)[:, None], nl.arange(512)[None, :]], dtype=np.float32, mask=None)
                        ' end loop i12 '
                    ' end loop i11 '
                ' end loop i10 '
            ' end loop i9 '
        ' end loop i2 '
    for i13 in nl.affine_range(12):
        for i14 in nl.sequential_range(4):
            for i15 in nl.affine_range(2):
                for i16 in nl.affine_range(5):
                    v17[i13, i14, i15, i16, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nl.load(v2[i16 + 5 * i15 + 10 * i14, nl.arange(128)[:, None], 1024 * i13 + nl.arange(1024)[None, :]], dtype=np.float32, mask=None)
                    ' end loop i16 '
                ' end loop i15 '
            for i17 in nl.affine_range(2):
                for i18 in nl.affine_range(4):
                    for i19 in nl.affine_range(2):
                        for i20 in nl.affine_range(4):
                            v18[i13, i14, i17, i18, i19, i20, nl.arange(128)[:, None, None], 128 * nl.arange(5)[None, :, None] + nl.arange(128)[None, None, :]] = nl.load(v1[4 * i17 + i18, i20, nl.arange(128)[:, None, None], 5 * i19 + 10 * i14 + nl.arange(5)[None, :, None], nl.arange(128)[None, None, :]], dtype=np.float32, mask=None)
                            for i21 in nl.affine_range(5):
                                v19[i13, i14, i17, i18, i19, i20, i21, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(v18[i13, i14, i17, i18, i19, i20, nl.arange(128)[:, None], 128 * i21 + nl.arange(128)[None, :]], v7[nl.arange(128)[:, None], nl.arange(128)[None, :]], is_stationary_onezero=False, is_moving_onezero=True, mask=None, is_transpose=True)
                                v20[i13, i17, i18, i14, i19, nl.arange(128)[:, None], i21, 128 * i20 + nl.arange(128)[None, :]] = nl.copy(v19[i13, i14, i17, i18, i19, i20, i21, nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32, mask=None)
                                ' end loop i21 '
                            ' end loop i20 '
                        ' end loop i19 '
                    for i22 in nl.affine_range(2):
                        for i23 in nl.affine_range(4):
                            for i24 in nl.affine_range(2):
                                for i25 in nl.affine_range(5):
                                    v21[i13, i14, i17, i18, i22, i23, nl.arange(128)[:, None], nl.arange(512)[None, :]] += nisa.nc_matmul(v17[i13, i14, i24, i25, nl.arange(128)[:, None], 128 * i23 + 512 * i22 + nl.arange(128)[None, :]], v20[i13, i17, i18, i14, i24, nl.arange(128)[:, None], i25, nl.arange(512)[None, :]], is_stationary_onezero=False, is_moving_onezero=False, mask=None)
                                    ' end loop i25 '
                                ' end loop i24 '
                            v22[i13, i22, i23, i17, i18, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.loop_reduce(v21[i13, i14, i17, i18, i22, i23, nl.arange(128)[:, None], nl.arange(512)[None, :]], op=np.add, loop_indices=[i14], mask=None, dtype=np.float32)
                            ' end loop i23 '
                        ' end loop i22 '
                    ' end loop i18 '
                ' end loop i17 '
            ' end loop i14 '
        for i26 in nl.affine_range(2):
            for i27 in nl.affine_range(4):
                for i28 in nl.affine_range(2):
                    for i29 in nl.affine_range(4):
                        v23[i26, i27, i13, i28, i29, nl.arange(128)[:, None], nl.arange(512)[None, :]] = nl.add(v22[i13, i28, i29, i26, i27, nl.arange(128)[:, None], nl.arange(512)[None, :]], v16[i13, i28, i29, i26, i27, nl.arange(128)[:, None], nl.arange(512)[None, :]], mask=None, dtype=np.float32)
                        ' end loop i29 '
                    for i30 in nl.affine_range(4):
                        for i31 in nl.affine_range(4):
                            v24[i13, i26, i27, i28, i30, i31, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(v23[i26, i27, i13, i28, i30, nl.arange(128)[:, None], 128 * i31 + nl.arange(128)[None, :]], v7[nl.arange(128)[:, None], nl.arange(128)[None, :]], is_stationary_onezero=False, is_moving_onezero=True, mask=None, is_transpose=True)
                            v25[i13, i26, i27, i28, i30, i31, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.copy(v24[i13, i26, i27, i28, i30, i31, nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32, mask=None)
                            nl.store(v5[4 * i26 + i27, i31, nl.arange(128)[:, None], i30 + 4 * i28 + 8 * i13, nl.arange(128)[None, :]], value=v25[i13, i26, i27, i28, i30, i31, nl.arange(128)[:, None], nl.arange(128)[None, :]], mask=None)
                            ' end loop i31 '
                        ' end loop i30 '
                    ' end loop i28 '
                ' end loop i27 '
            ' end loop i26 '
        ' end loop i13 '
    return v5