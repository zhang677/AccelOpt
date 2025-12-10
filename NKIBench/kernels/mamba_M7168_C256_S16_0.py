import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
import numpy as np

@nki.jit
def kernel(delta, u, a, b, c):
    channels, seq_len = delta.shape
    output = nl.ndarray((channels, seq_len), dtype=delta.dtype,
                        buffer=nl.shared_hbm)

    _, state_size = a.shape

    # We can relax this using mask paramters in all the NKI API calls
    assert channels % 128 == 0

    # Map channels to the partition dimension
    # Tile channels to comply with NKI tile size constraints
    channel_psize = nl.tile_size.pmax
    n_channel_tile = channels // channel_psize

    # partial accumulated scanC result with processed states
    scanC_accum = nl.zeros((n_channel_tile, nl.par_dim(channel_psize), seq_len), dtype=delta.dtype)

    # Second outer loop with state_size, partial parallel
    for i_state in nl.affine_range(state_size):

        # Inner loop: tiling channels
        for i_channel_tile in nl.affine_range(n_channel_tile):
            channel_start = i_channel_tile * channel_psize

            # Load the relevant tile from delta and A
            delta_i = nl.load(delta[channel_start:channel_start+channel_psize, 0:seq_len])
            A_i = nl.load(a[channel_start:channel_start+channel_psize, i_state])

            # Step 1&2: Element-wise multiplication of delta_i and A_i and then exponential
            deltaA = nisa.activation(op=nl.exp, data=delta_i, scale=A_i)

            # Load the relevant tile from u and B
            u_i = nl.load(u[channel_start:channel_start+channel_psize, 0:seq_len])
            B_i = nl.load(b[i_state:i_state+1, 0:seq_len])

            # Step 3: Element-wise multiplication of delta_i, B_i and u_i
            deltaU = nisa.tensor_tensor(delta_i, u_i, op=nl.multiply)
            B_i_bcast = B_i.broadcast_to((channel_psize, seq_len))
            deltaBu = nisa.tensor_tensor(deltaU, B_i_bcast, op=nl.multiply)

            # Step 4: Associative scan between deltaA and deltaBu
            scan_res = nki.isa.tensor_tensor_scan(deltaA, deltaBu, initial=0,
                    op0=np.multiply, op1=np.add)

            # Load the relevant tile from C
            C_i = nl.load(c[i_state:i_state+1, 0:seq_len])

            # Step 5: Element-wise multiplication of scan_res and C_i
            C_i_bcast = C_i.broadcast_to((channel_psize, seq_len))
            scanC = nisa.tensor_tensor(scan_res, C_i_bcast, op=nl.multiply)

            # Step 6: Accumulation of scanC along state_size dimension
            scanC_accum[i_channel_tile, 0:channel_psize, 0:seq_len] += scanC

    # Store scanC_accum for a single batch to output
    for i_channel_tile in nl.affine_range(n_channel_tile):
        channel_start = i_channel_tile * channel_psize
        nl.store(output[channel_start:channel_start+channel_psize, 0:seq_len],
                scanC_accum[i_channel_tile, 0:channel_psize, 0:seq_len])

    return output