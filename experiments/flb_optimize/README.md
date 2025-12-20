Optimize the workloads with baseline ~20ms; Collect the correct triton implementations

```
python select_traces.py
python create_files.py # Create jsons for all selected traces
python create_tables.py # Only get partial table
```

Copy the blob tensors to checkpoints
```
cp -r /home/ubuntu/flashinfer-trace/blob /home/ubuntu/AccelOpt/experiments/flb_interface/checkpoints
```