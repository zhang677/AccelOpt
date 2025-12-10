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
  tile_size_S = 512
  i_dh = nl.arange(d_head)[:, None]
  i_S = nl.arange(tile_size_S)[None, :]
  for i_S_offset in range(0, S, tile_size_S):
    # Load input tensor.
    x_in_sb = nl.ndarray((d_head, tile_size_S), dtype=x_in.dtype, buffer=nl.sbuf)
    x_in_sb[i_dh, i_S] = nl.load(x_in[i_dh, i_S + i_S_offset])

    # Pack cos and sin on partition dimension to save sbuf usage.
    sb_coeff = nl.ndarray((d_head, tile_size_S), dtype=x_in.dtype, buffer=nl.sbuf)
    sb_cos = sb_coeff[i_upper, i_S]
    sb_sin = sb_coeff[i_lower, i_S]
    sb_cos = nl.load(cos[i_upper, i_S + i_S_offset])
    sb_sin = nl.load(sin[i_upper, i_S + i_S_offset])

    x_out_sb = nl.ndarray((d_head, tile_size_S), dtype=x_in.dtype, buffer=nl.sbuf)

    # Inlined RoPE_sbuf implementation
    sb_e = x_in_sb[i_upper, i_S]
    # copy to make sure tensortensor have both inputs with the same base partition
    sb_o = nl.copy(x_in_sb[i_lower, i_S])

    e_cos_sin = nl.ndarray((d_head, tile_size_S), dtype=x_in_sb.dtype, buffer=nl.sbuf)
    e_cos = e_cos_sin[i_upper, i_S]
    e_sin = e_cos_sin[i_lower, i_S]

    o_cos_sin = nl.ndarray((d_head, tile_size_S), dtype=x_in_sb.dtype, buffer=nl.sbuf)
    o_cos = o_cos_sin[i_upper, i_S]
    o_sin = o_cos_sin[i_lower, i_S]

    e_cos = nisa.tensor_tensor(sb_e, sb_cos, nl.multiply)
    o_cos = nisa.tensor_tensor(sb_o, sb_cos, nl.multiply)
    e_sin = nisa.tensor_tensor(sb_e, sb_sin, nl.multiply)
    o_sin = nisa.tensor_tensor(sb_o, sb_sin, nl.multiply)

    x_out_sb[i_upper, i_S] = nisa.tensor_tensor(e_cos, o_sin, nl.subtract) # even * cos -  odd * sin
    x_out_sb[i_lower, i_S] = nisa.tensor_tensor(o_cos, e_sin, nl.add)      #  odd * cos + even * sin

    # Store output tensor.
    nl.store(x_out[i_dh, i_S + i_S_offset], x_out_sb)
  return x_out