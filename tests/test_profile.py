from accelopt.kernel_wrapper import NKIKernel
import json

def test_single_profile(program_path, base_numpy_path):
    nki_kernel = NKIKernel(program_path, base_numpy_path)
    save_fields_path = "../prompts/profile_list.json"
    with open(save_fields_path, "r") as f:
        save_fields = json.load(f)
    nki_kernel.profile(save_fields)
    return nki_kernel.res

if __name__ == "__main__":
    program_path = "../NKIBench/kernels/adamw_M10944_N2048_0.py"
    base_numpy_path = "../NKIBench/reference/adamw_M10944_N2048_numpy_1.py"
    result = test_single_profile(program_path, base_numpy_path)
    print(result)