EXP_DATE=$1
EXP_BASE_DIR=$2
NC_ID=$3
MAX_THRESHOLD=$4
MIN_THRESHOLD=$5
TOPK=$6
TOPK_CANDIDATES=$7
PROJECT_NAME=$8
REL_TOL=$9
ORG_NAME="${10}"

EXP_DIR=$EXP_BASE_DIR/$EXP_DATE

cd $EXP_DIR

logfire projects use $PROJECT_NAME --org $ORG_NAME
LOGFIRE_ENV_NAME=$(cat $EXP_DIR/logfire_env_name.txt)
export LOGFIRE_ENVIRONMENT=$LOGFIRE_ENV_NAME
echo "LOGFIRE_ENVIRONMENT: $LOGFIRE_ENVIRONMENT"

timestamp() { date -u +"%Y-%m-%dT%H:%M:%S.%6NZ"; }

start_time="$(timestamp)"

mkdir -p $EXP_DIR/rewrites
EXEC="$ACCELOPT_BASE_DIR/scripts/rewrites_selection.py"
EXECUTOR_RESULTS_PATH="$EXP_DIR/candidates/last_iteration_executor_results.json"
BASE_PROMPT_PATH="$ACCELOPT_BASE_DIR/prompts/summarizer_prompts/base_prompt.txt"
USER_TEMPLATE_PATH="$ACCELOPT_BASE_DIR/prompts/summarizer_prompts/user_prompt_template.txt"
OUTPUT_LIST_PATH="$EXP_DIR/rewrites/rewrites_selection_output_list.json"
OUTPUT_SPEEDUPS_PATH="$EXP_DIR/rewrites/rewrites_selection_output_speedups.json"
OUTPUT_PLAN_IDS_PATH="$EXP_DIR/rewrites/rewrites_selection_output_plan_ids.json"
MODEL_CONFIG_PATH="$EXP_BASE_DIR/configs/summarizer_config.json"
python $EXEC --executor_results_path $EXECUTOR_RESULTS_PATH --base_prompt_path $BASE_PROMPT_PATH --user_template_path $USER_TEMPLATE_PATH --output_list_path $OUTPUT_LIST_PATH --max_threshold $MAX_THRESHOLD --min_threshold $MIN_THRESHOLD --topk $TOPK --output_plan_ids_path $OUTPUT_PLAN_IDS_PATH --output_speedups_path $OUTPUT_SPEEDUPS_PATH --model_config_path $MODEL_CONFIG_PATH

end_time="$(timestamp)"
printf "%s,%s\n" "${start_time}" "${end_time}" >> "${EXP_DIR}/rewrites_selection_start_end_time.txt"

start_time="$(timestamp)"

EXEC="$ACCELOPT_BASE_DIR/scripts/select_candidates.py"
EXECUTOR_RESULTS_PATH="$EXP_DIR/candidates/last_iteration_executor_results.json"
OUTPUT_BASE_PATH="$EXP_DIR/candidates"
python $EXEC --executor_results_path $EXECUTOR_RESULTS_PATH --output_base_path $OUTPUT_BASE_PATH --topk $TOPK_CANDIDATES

end_time="$(timestamp)"
printf "%s,%s\n" "${start_time}" "${end_time}" >> "${EXP_DIR}/select_candidates_start_end_time.txt"


start_time="$(timestamp)"

EXEC="$ACCELOPT_BASE_DIR/scripts/sequential_profile.py"
CANDIDATES_PATH="$EXP_DIR/candidates/candidates.csv"
SAVE_FIELDS_PATH="$ACCELOPT_BASE_DIR/prompts/profile_list.json"
OUTPUT_PATH="$EXP_DIR/candidates/profile_results.csv"

python $EXEC --candidates_path $CANDIDATES_PATH --save_fields_path $SAVE_FIELDS_PATH --output_path $OUTPUT_PATH --nc_id $NC_ID --rel_tol $REL_TOL

end_time="$(timestamp)"
printf "%s,%s\n" "${start_time}" "${end_time}" >> "${EXP_DIR}/profile_candidates_start_end_time.txt"

cd $EXP_BASE_DIR