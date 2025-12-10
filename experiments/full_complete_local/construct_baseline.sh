COLLECT_EXEC="$ACCELOPT_BASE_DIR/scripts/collect_candidates.py"
SUMMARY_PATH="$ACCELOPT_BASE_DIR/NKIBench/summary.json"
CANDIDATES_PATH="candidates.csv"
PROFILE_PATH="profile_results.csv"
SAVE_FIELDS_PATH="$ACCELOPT_BASE_DIR/prompts/profile_list.json"
NC_ID=$1
python $COLLECT_EXEC --summary_path $SUMMARY_PATH --output_candidates_path $CANDIDATES_PATH --output_profile_path $PROFILE_PATH --save_fields_path $SAVE_FIELDS_PATH --nc_id $NC_ID --mode construct