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
    v5 = nl.ndarray((nl.par_dim(128), 128), dtype=np.float32, name='identity_local_77', buffer=nl.sbuf)
    v6 = nl.ndarray((16, 4, nl.par_dim(64), 1024), dtype=np.float32, name='rhs_local_45', buffer=nl.sbuf)
    v7 = nl.ndarray((4, 16, 4, 8, nl.par_dim(128), 64), dtype=np.float32, name='', buffer=nl.sbuf)
    v8 = nl.zeros((16, 4, 4, 8, nl.par_dim(64), 128), dtype=np.float32, name='34.73', buffer=nl.psum, lazy_initialization=True)
    v9 = nl.ndarray((4, 16, 4, nl.par_dim(64), 1024), dtype=np.float32, name='', buffer=nl.sbuf)
    v10 = nl.zeros((16, 4, 4, 8, 2, nl.par_dim(128), 512), dtype=np.float32, name='', buffer=nl.psum, lazy_initialization=True)
    v11 = nl.ndarray((16, 4, 8, 4, nl.par_dim(128), 1024), dtype=np.float32, name='', buffer=nl.sbuf)
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
                        v11[i0, i2, i4, i1, nl.arange(128)[:, None], 512 * i5 + nl.arange(512)[None, :]] = nl.copy(v10[i0, i1, i2, i4, i5, nl.arange(128)[:, None], nl.arange(512)[None, :]], dtype=np.float32, mask=None)
                        ' end loop i5 '
                    nl.store(v3[i0, i4 + 8 * i2, nl.arange(128)[:, None], 1024 * i1 + nl.arange(1024)[None, :]], value=v11[i0, i2, i4, i1, nl.arange(128)[:, None], nl.arange(1024)[None, :]], mask=None)
                    ' end loop i4 '
                ' end loop i2 '
            ' end loop i1 '
        ' end loop i0 '
    return v3