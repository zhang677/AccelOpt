Beam Search + Optimization Memory + Local served gpt-oss-120b

Run:
```
export ACCELOPT_BASE_DIR={Directory of AccelOpt}
logfire projects use "{Project Name}" --org "{Org Name}" # If the $PROJECT_NAME doesn't exist, then create it first: logfire projects create "{Project Name}" --org "{Org Name}"
./construct_baseline.sh 0
python create_folders.py --project_name "{Project Name}" --org_name "{Org Name}" --exp_date_base "11-11-09-04"
./create_all_sessions.sh "11-11-09-04"
./kill_all_sessions.sh "11-11-09-04" # Clean tmux sessions after the experiments
python plot_trace.py # Plot the trace
python calculate_percentage_of_peak.py # Calculate the percentage of peak throughput
python clean_checkpoints.py --chkpt_path "../checkpoints/11-11-09-04"
python resume_folders.py --project_name "{Project Name}" --org_name "{Org Name}" --exp_date_base "11-11-09-04"
./create_all_resume_sessions.sh "11-11-09-04"
./kill_all_resume_sessions.sh "11-11-09-04"
```

