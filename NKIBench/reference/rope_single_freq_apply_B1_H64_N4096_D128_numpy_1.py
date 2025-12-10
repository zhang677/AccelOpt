import numpy as np

B = 1
H = 64
N = 4096
D = 128

def get_inputs():
    x = np.random.normal(loc=0, scale=1.0, size=(D, B*H*N)).astype(np.float32)
    freqs_cos = np.random.normal(loc=0, scale=1.0, size=(D // 2, B*H*N)).astype(np.float32)
    freqs_sin = np.random.normal(loc=0, scale=1.0, size=(D // 2, B*H*N)).astype(np.float32)
    return [x, freqs_cos, freqs_sin]

def forward(x, freqs_cos, freqs_sin):
    half_h = D // 2
    x0 = x[:half_h, :]
    x1 = x[half_h:, :]
    x_out_0 = x0 * freqs_cos - x1 * freqs_sin
    x_out_1 = x0 * freqs_sin + x1 * freqs_cos
    x_out = np.concatenate([x_out_0, x_out_1], axis=0)
    return x_out

def transform_to_nki_inputs(inputs):
    return inputs

def transform_nki_outputs(k_res, ref):
    return (k_res,)