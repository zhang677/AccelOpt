from pcl.remote_utils import NKIKernel
import os
import tempfile
import json

baseline_percentage = 0.28358070488082826
baseline_latency = 2.888772

def _write_temp_kernel(code: str) -> str:
    fd, temp_path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(
            "import numpy as np\n"
            "import neuronxcc.nki as nki\n"
            "import neuronxcc.nki.language as nl\n"
            "import neuronxcc.nki.typing as nt\n"
            "import neuronxcc.nki.isa as nisa\n"
            "from neuronxcc.nki import trace\n"
            "from neuronxcc.nki.language import par_dim\n\n"
            f"{code}\n"
        )
    return temp_path

save_fields_path = "../prompts/profile_list.json"

with open(save_fields_path, "r") as f:
    save_fields = json.load(f)


os.environ["NEURON_RT_VISIBLE_CORES"] = '0'
nki_kernel_paths = [
    "mamba_v1.py",
    "mamba_v2.py",
    "mamba_v3.py",
    "mamba_optimized.py",
]

numpy_path = os.path.abspath("../NKIBench/reference/mamba_M7168_C256_S16_numpy_1.py")
nki_kernel_paths = [os.path.abspath(p) for p in nki_kernel_paths]
for nki_kernel_path in nki_kernel_paths:
    with open(nki_kernel_path, "r") as f:
        kernel_code = f.read()
    temp_kernel_path = _write_temp_kernel(kernel_code)
    kernel = NKIKernel(temp_kernel_path, numpy_path)
    kernel.rel_tol = 3e-5
    kernel.profile(save_fields=save_fields)
    print(kernel.res)
    os.remove(temp_kernel_path)
    latency = kernel.res.metadata['latency']
    print("Kernel path:", nki_kernel_path)
    print("Latency & Percentage:", latency, baseline_percentage * baseline_latency / latency)