Optimize the workloads with baseline ~20ms; Collect the correct triton implementations

```
python select_traces.py
python create_files.py # Create jsons for all selected traces
python create_tables.py # Only get partial table if using all selected traces in create_files.py
```

Copy the blob tensors to checkpoints
```
cp -r /home/ubuntu/flashinfer-trace/blob /home/ubuntu/AccelOpt/experiments/flb_interface/checkpoints
```

The checkpoint format:
definition_path,solution_path,workload_path,trace_path

definition_path: Definition (problem + values + case_name)
solution_path: Solution (kernel / service_name)
workload_path: Trace (task)
trace_path: Trace (profile)

Leave the displayed_profiles_path for the future ncu integration