import sys
import numpy as np
import re
from pathlib import Path
from pydantic import BaseModel, Field

class KernelProperties(BaseModel):
    """
    Single Kernel Execution
    """
    compiled: bool = False
    correct: bool = False
    runnable: bool = False
    metadata: dict = Field(default_factory=dict)

# Function to load module from file path
def load_module_from_path(file_path):
    parent_dir = str(Path(file_path).parent)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    import importlib.util
    spec = importlib.util.spec_from_file_location("module", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def l2norm_allclose(v_k, v_r, rel_tol=1e-5):
    return np.linalg.norm((v_k - v_r).astype(np.float64)) < rel_tol * np.linalg.norm(v_r.astype(np.float64))

def check_correctness_numpy(output_nki, output_task, res, rel_tol=2e-5):
    # output_nki is a list
    # output_task is a tuple or a single array
    if not isinstance(output_task, tuple):
        output_task_tuple = (output_task,)
    else:
        output_task_tuple = output_task
    is_correct = True
    if len(output_nki) != len(output_task_tuple):
        res.metadata.setdefault("correctness_error", []).append(
            f"Num outputs mismatch: nki={len(output_nki)} vs ref={len(output_task_tuple)}"
        )
        res.correct = False
        return
    for i, (v_k, v_r) in enumerate(zip(output_nki, output_task_tuple)):
        if hasattr(v_r, "shape") and hasattr(v_k, "shape"):
            if v_k.shape != v_r.shape:
                res.metadata.setdefault("correctness_error", []).append(f"Output {i} shape mismatch, expected {v_r.shape}, got {v_k.shape}; ")
                is_correct = False
            if not l2norm_allclose(v_k, v_r, rel_tol=rel_tol):
                max_diff = np.amax(np.abs(v_k - v_r))
                avg_diff = np.mean(np.abs(v_k - v_r))
                max_rel_diff = np.amax(np.abs(v_k - v_r) / np.abs(v_r))
                l2norm_diff = np.linalg.norm((v_k - v_r).astype(np.float64))
                l2norm_ref = np.linalg.norm(v_r.astype(np.float64))
                l2norm_rel_diff = l2norm_diff / l2norm_ref
                res.metadata.setdefault("correctness_error", []).append(f"Output {i} value mismatch, max diff {max_diff:.6f}, avg diff {avg_diff:.6f}, max rel diff {max_rel_diff:.6f}, l2norm diff {l2norm_diff:.6f}, l2norm ref {l2norm_ref:.6f}, l2norm rel diff {l2norm_rel_diff:.6f}")
                is_correct = False
        else:
            # abs_diff = np.abs(v_k - v_r)
            if np.issubdtype(type(v_r), np.floating) or np.issubdtype(type(v_k), np.floating):
                if not l2norm_allclose(v_k, v_r, rel_tol=rel_tol):
                    res.metadata.setdefault("correctness_error", []).append(f"Output {i} value mismatch, expected {v_r}, got {v_k};")
                    is_correct = False
            else:
                if v_k != v_r:
                    res.metadata.setdefault("correctness_error", []).append(f"Output {i} value mismatch, expected {v_r}, got {v_k}; ")
                    is_correct = False
    res.correct = is_correct

def check_precision_and_correctness(program_path, output_nki, output_task, res, rel_tol):    
    with open(program_path, 'r') as f:
        program_code = f.read()
    # Remove all the comments
    program_code = re.sub(r'#.*', '', program_code)
    # If "bfloat16" or "float16" is used
    if "float16" in program_code:
        res.metadata["correctness_error"] = "Float16 is used in the program."
        res.correct = False
        return
    check_correctness_numpy(output_nki, output_task, res, rel_tol=rel_tol)