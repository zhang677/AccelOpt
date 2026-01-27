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

    # Sequence tiling to reduce buffer size
    seq_tile_size = 512  # Process sequence in chunks of 512
    n_seq_tile = seq_len // seq_tile_size  # Assuming seq_len is divisible by seq_tile_size
    assert seq_len % seq_tile_size == 0, "seq_len must be divisible by seq_tile_size"

    # Initialize scan state for maintaining dependencies across sequence tiles
    scan_state = nl.zeros((n_channel_tile, nl.par_dim(channel_psize), state_size), dtype=delta.dtype)

    # Sequential processing of sequence tiles (due to scan dependencies)
    for i_seq_tile in nl.sequential_range(n_seq_tile):
        seq_start = i_seq_tile * seq_tile_size
        seq_end = seq_start + seq_tile_size
        
        # Reduced size accumulation buffer for current sequence tile
        scanC_accum = nl.zeros((n_channel_tile, nl.par_dim(channel_psize), seq_tile_size), dtype=delta.dtype)

        # Swapped loop order: channels outer, state inner for better data reuse
        for i_channel_tile in nl.affine_range(n_channel_tile):
            channel_start = i_channel_tile * channel_psize
            
            # Load delta and u once per channel tile (moved outside state loop)
            delta_i = nl.load(delta[channel_start:channel_start+channel_psize, seq_start:seq_end])
            u_i = nl.load(u[channel_start:channel_start+channel_psize, seq_start:seq_end])
            
            # Inner loop with state_size
            for i_state in nl.affine_range(state_size):
                # Load the relevant tile from A
                A_i = nl.load(a[channel_start:channel_start+channel_psize, i_state])

                # Step 1&2: Element-wise multiplication of delta_i and A_i and then exponential
                deltaA = nisa.activation(op=nl.exp, data=delta_i, scale=A_i)

                # Load the relevant tile from B
                B_i = nl.load(b[i_state:i_state+1, seq_start:seq_end])

                # Step 3: Element-wise multiplication of delta_i, B_i and u_i
                deltaU = nisa.tensor_tensor(delta_i, u_i, op=nl.multiply)
                B_i_bcast = B_i.broadcast_to((channel_psize, seq_tile_size))
                deltaBu = nisa.tensor_tensor(deltaU, B_i_bcast, op=nl.multiply)

                # Step 4: Associative scan between deltaA and deltaBu with proper initial state
                initial_state = scan_state[i_channel_tile, 0:channel_psize, i_state] if i_seq_tile > 0 else 0
                scan_res = nki.isa.tensor_tensor_scan(deltaA, deltaBu, initial=initial_state,
                        op0=np.multiply, op1=np.add)

                # Update scan state for next sequence tile (take last element)
                if i_seq_tile < n_seq_tile - 1:
                    # Extract the last element of scan_res for this state
                    last_scan_val = scan_res[:, seq_tile_size-1:seq_tile_size]
                    scan_state[i_channel_tile, 0:channel_psize, i_state:i_state+1] = last_scan_val

                # Load the relevant tile from C
                C_i = nl.load(c[i_state:i_state+1, seq_start:seq_end])

                # Step 5: Element-wise multiplication of scan_res and C_i
                C_i_bcast = C_i.broadcast_to((channel_psize, seq_tile_size))
                scanC = nisa.tensor_tensor(scan_res, C_i_bcast, op=nl.multiply)

                # Step 6: Accumulation of scanC along state_size dimension
                scanC_accum[i_channel_tile, 0:channel_psize, 0:seq_tile_size] += scanC

        # Store scanC_accum for current sequence tile directly to output
        for i_channel_tile in nl.affine_range(n_channel_tile):
            channel_start = i_channel_tile * channel_psize
            nl.store(output[channel_start:channel_start+channel_psize, seq_start:seq_end],
                    scanC_accum[i_channel_tile, 0:channel_psize, 0:seq_tile_size])

    return output