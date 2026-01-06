# Test profiling one kernel
```
python test_profile.py
```

# Test profiling many kernels
```
./construct_baseline.sh
```
This will generate a candidates.csv and a profile_results.csv

# Test signoz log, query, and delete
```
./run_test_signoz.sh
```
Knobs for query and delete in the test_signoz.py. Be really careful with the delete!