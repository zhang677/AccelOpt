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
    v3 = nl.ndarray((32, 128, 12288), dtype=np.float32, buffer=nl.shared_hbm)
    v4 = nl.shared_constant(np.identity(128, dtype=np.float32))
    v5 = nl.ndarray((nl.par_dim(128), 128), dtype=np.float32, name='identity_local_80', buffer=nl.sbuf)
    v6 = nl.ndarray((12, 5, 8, nl.par_dim(128), 1024), dtype=np.float32, name='rhs_local_49', buffer=nl.sbuf)
    v7 = nl.ndarray((12, 32, 5, nl.par_dim(128), 1024), dtype=np.float32, name='', buffer=nl.sbuf)
    v8 = nl.zeros((12, 32, 5, 8, nl.par_dim(128), 128), dtype=np.float32, name='38.76', buffer=nl.psum, lazy_initialization=True)
    v9 = nl.ndarray((12, 32, 5, nl.par_dim(128), 8, 128), dtype=np.float32, name='', buffer=nl.sbuf)
    v10 = nl.zeros((12, 32, 2, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.psum, lazy_initialization=True)
    v11 = nl.ndarray((32, 12, nl.par_dim(128), 1024), dtype=np.float32, name='', buffer=nl.sbuf)
    v5[nl.arange(128)[:, None], nl.arange(128)[None, :]] = nl.load(v4[nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32, mask=None)
    for i0 in nl.affine_range(12):
        for i1 in nl.affine_range(5):
            for i2 in nl.affine_range(8):
                v6[i0, i1, i2, nl.arange(128)[:, None], nl.arange(1024)[None, :]] = nl.load(v2[i2 + 8 * i1, nl.arange(128)[:, None], nl.arange(1024)[None, :] + 1024 * i0], dtype=np.float32, mask=None)
                ' end loop i2 '
            ' end loop i1 '
        for i3 in nl.affine_range(32):
            for i4 in nl.affine_range(5):
                v7[i0, i3, i4, nl.arange(128)[:, None, None], 128 * nl.arange(8)[None, :, None] + nl.arange(128)[None, None, :]] = nl.load(v1[i3, nl.arange(128)[:, None, None], nl.arange(8)[None, :, None] + 8 * i4, nl.arange(128)[None, None, :]], dtype=np.float32, mask=None)
                for i5 in nl.affine_range(8):
                    v8[i0, i3, i4, i5, nl.arange(128)[:, None], nl.arange(128)[None, :]] = nisa.nc_matmul(v7[i0, i3, i4, nl.arange(128)[:, None], 128 * i5 + nl.arange(128)[None, :]], v5[nl.arange(128)[:, None], nl.arange(128)[None, :]], is_stationary_onezero=False, is_moving_onezero=True, mask=None, is_transpose=True)
                    v9[i0, i3, i4, nl.arange(128)[:, None], i5, nl.arange(128)[None, :]] = nl.copy(v8[i0, i3, i4, i5, nl.arange(128)[:, None], nl.arange(128)[None, :]], dtype=np.float32, mask=None)
                    ' end loop i5 '
                ' end loop i4 '
            for i6 in nl.affine_range(2):
                for i7 in nl.affine_range(5):
                    for i8 in nl.affine_range(8):
                        v10[i0, i3, i6, nl.arange(128)[:, None], nl.arange(512)[None, :]] += nisa.nc_matmul(v9[i0, i3, i7, nl.arange(128)[:, None], i8, nl.arange(128)[None, :]], v6[i0, i7, i8, nl.arange(128)[:, None], 512 * i6 + nl.arange(512)[None, :]], is_stationary_onezero=False, is_moving_onezero=False, mask=None)
                        ' end loop i8 '
                    ' end loop i7 '
                v11[i3, i0, nl.arange(128)[:, None], 512 * i6 + nl.arange(512)[None, :]] = nl.copy(v10[i0, i3, i6, nl.arange(128)[:, None], nl.arange(512)[None, :]], dtype=np.float32, mask=None)
                ' end loop i6 '
            nl.store(v3[i3, nl.arange(128)[:, None], nl.arange(1024)[None, :] + 1024 * i0], value=v11[i3, i0, nl.arange(128)[:, None], nl.arange(1024)[None, :]], mask=None)
            ' end loop i3 '
        ' end loop i0 '
    return v3