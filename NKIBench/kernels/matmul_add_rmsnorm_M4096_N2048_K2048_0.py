import neuronxcc.nki as nki
import neuronxcc.nki.language as nl


@nki.jit
def kernel(x_tensor, w_tensor, eps, z_tensor, g_tensor):
  M, K = x_tensor.shape
  K_, N = w_tensor.shape
  # Make sure shapes match
  assert N == g_tensor.shape[0]
  assert K_ == K
  assert N == w_tensor.shape[1]
  assert M == z_tensor.shape[0]
  assert N == z_tensor.shape[1]

  TILE_M = 128
  TILE_K = 128
  TILE_N = 512 # nl.tile_size.gemm_moving_fmax
  ix = nl.arange(TILE_M)[:, None]
  iw = nl.arange(1)[:, None]
  iy = nl.arange(N)[None, :]
  iz = nl.arange(TILE_N)[None, :]
  ik = nl.arange(K)[None, :]

  result = nl.ndarray((M, N), dtype=x_tensor.dtype, buffer=nl.shared_hbm)
  g_tile = nl.load(g_tensor.reshape((1, N))[iw, iy])
  for i in nl.affine_range(M // TILE_M):
    rmsnorm_in_tile = nl.ndarray((TILE_M, N), dtype=x_tensor.dtype, buffer=nl.sbuf)
    x_tiles = nl.load(x_tensor[i * TILE_M + ix, ik])
    for n in nl.affine_range(N // TILE_N):
      res_psum = nl.zeros((TILE_M, TILE_N), nl.float32, buffer=nl.psum)
      for k in nl.affine_range(K // TILE_K):
        w_tile = nl.load(w_tensor[k * TILE_K: (k + 1) * TILE_K, n * TILE_N: (n + 1) * TILE_N])
        res_psum += nl.matmul(x_tiles[:, k * TILE_K: (k + 1) * TILE_K], w_tile)
      res_sb = nl.copy(res_psum, dtype=result.dtype)
      rmsnorm_in_tile[ix, n * TILE_N + iz] = res_sb
    z_tile = nl.load(z_tensor[i * TILE_M + ix, iy])
    a_tile = nl.add(rmsnorm_in_tile, z_tile)
    in_square = nl.square(a_tile)
    square_sum = nl.sum(in_square, axis=[1])
    mean = square_sum / N  # Changed from K to N - normalize over output dimension
    mean = nl.add(mean, eps)
    rms_reciprocal = nl.rsqrt(mean)
    rmsnorm_out_tile = nl.multiply(a_tile, rms_reciprocal)
    g_bcast = g_tile.broadcast_to((TILE_M, N))
    rmsnorm_out_tile[...] = nl.multiply(rmsnorm_out_tile, g_bcast)
    nl.store(result[i * TILE_M + ix, iy], value=rmsnorm_out_tile)
  return result 