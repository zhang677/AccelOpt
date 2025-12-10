from accelopt.kernel_wrapper import NKIKernel
import json
import os
import pandas as pd
import tempfile

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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates_path", type=str, required=True)
    parser.add_argument("--save_fields_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--nc_id", type=int, required=True)
    parser.add_argument("--rel_tol", type=float, required=True)
    args = parser.parse_args()
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(args.nc_id)
    with open(args.save_fields_path, "r") as f:
        save_fields = json.load(f)
    df = pd.read_csv(args.candidates_path)
    row_data_list = df.to_dict(orient="records")
    output_rows_data = []
    for row_data in row_data_list:
        with open(row_data["kernel"], "r") as f:
            kernel_code = f.read()
        temp_kernel_path = _write_temp_kernel(kernel_code)
        nki_kernel = NKIKernel(temp_kernel_path, row_data["task"])
        nki_kernel.rel_tol = args.rel_tol
        nki_kernel.profile(save_fields)
        profile_data = {"profile": json.dumps(nki_kernel.res.metadata)}
        output_rows_data.append({**row_data, **profile_data})
        if temp_kernel_path:
            os.remove(temp_kernel_path)
    output_df = pd.DataFrame(output_rows_data)
    output_df.to_csv(args.output_path, index=False)
        

