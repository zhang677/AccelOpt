import neuronxcc.nki as nki
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl

@nki.jit
def kernel(x_in, cos, sin):
  """
  Applies rotary position embeddings.
  Expected layout:
  x_in: [d_head, S]
  cos, sin: [d_head // 2, S]

  This implementation uses first and second halves i.e.,

    result[:d_head/2] = embedding[:d_head/2] * cos - embedding[d_head/2:] * sin
    result[d_head/2:]  = embedding[d_head/2:] * cos + embedding[:d_head/2] * sin
  """

  d_head, S = x_in.shape
  half_d = d_head // 2
  assert d_head <= 128
  assert tuple(cos.shape) == (half_d, S)
  assert cos.shape == sin.shape

  x_out = nl.ndarray((d_head, S), dtype=x_in.dtype, buffer=nl.shared_hbm)
  # Indices for selecting upper, lower partitions.
  i_upper = nl.arange(half_d)[:, None]
  i_lower = i_upper + half_d

  # Tile along the S dimension.
  tile_size_S = 2048
  i_S = nl.arange(tile_size_S)[None, :]
  
  for i_S_offset in range(0, S, tile_size_S):
    # Load the required slices directly into the tensor ops
    e_cos  = nisa.tensor_tensor(
                nl.load(x_in[i_upper,  i_S + i_S_offset]),
                nl.load(cos[i_upper,  i_S + i_S_offset]),
                nl.multiply)
    o_sin  = nisa.tensor_tensor(
                nl.load(x_in[i_lower, i_S + i_S_offset]),
                nl.load(sin[i_upper,  i_S + i_S_offset]),
                nl.multiply)
    e_sin  = nisa.tensor_tensor(
                nl.load(x_in[i_upper,  i_S + i_S_offset]),
                nl.load(sin[i_upper,  i_S + i_S_offset]),
                nl.multiply)
    o_cos  = nisa.tensor_tensor(
                nl.load(x_in[i_lower, i_S + i_S_offset]),
                nl.load(cos[i_upper,  i_S + i_S_offset]),
                nl.multiply)

    # Store the fused results directly to HBM
    nl.store(
        x_out[i_upper, i_S + i_S_offset],
        nisa.tensor_tensor(e_cos, o_sin, nl.subtract))      # even * cos – odd * sin
    nl.store(
        x_out[i_lower, i_S + i_S_offset],
        nisa.tensor_tensor(o_cos, e_sin, nl.add))           #  odd * cos + even * sin

  return x_out