EXP_DATE=$1
EXP_BASE_DIR=$2
MAX_THRESHOLD=$3
MIN_THRESHOLD=$4
TOPK=$5
TOPK_CANDIDATES=$6

EXP_DIR=$EXP_BASE_DIR/$EXP_DATE

cd $EXP_DIR


LOG_ENV_NAME=$(cat $EXP_DIR/log_env_name.txt)
export OTEL_SERVICE_NAME=$LOG_ENV_NAME 
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
export OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=true
export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true
echo "OTEL_SERVICE_NAME: $OTEL_SERVICE_NAME"

timestamp() { date -u +"%Y-%m-%dT%H:%M:%S.%6NZ"; }

start_time="$(timestamp)"

mkdir -p $EXP_DIR/rewrites
EXEC="$ACCELOPT_BASE_DIR/scripts/fib/rewrites_selection.py"
EXECUTOR_RESULTS_PATH="$EXP_DIR/candidates/last_iteration_executor_results.json"
BASE_PROMPT_PATH="$ACCELOPT_BASE_DIR/prompts/flb/summarizer_prompts/base_prompt.txt"
USER_TEMPLATE_PATH="$ACCELOPT_BASE_DIR/prompts/flb/summarizer_prompts/user_prompt_template.txt"
OUTPUT_LIST_PATH="$EXP_DIR/rewrites/rewrites_selection_output_list.json"
OUTPUT_SPEEDUPS_PATH="$EXP_DIR/rewrites/rewrites_selection_output_speedups.json"
OUTPUT_PLAN_IDS_PATH="$EXP_DIR/rewrites/rewrites_selection_output_plan_ids.json"
MODEL_CONFIG_PATH="$EXP_BASE_DIR/configs/summarizer_config.json"
opentelemetry-instrument python $EXEC --executor_results_path $EXECUTOR_RESULTS_PATH --base_prompt_path $BASE_PROMPT_PATH --user_template_path $USER_TEMPLATE_PATH --output_list_path $OUTPUT_LIST_PATH --max_threshold $MAX_THRESHOLD --min_threshold $MIN_THRESHOLD --topk $TOPK --output_plan_ids_path $OUTPUT_PLAN_IDS_PATH --output_speedups_path $OUTPUT_SPEEDUPS_PATH --model_config_path $MODEL_CONFIG_PATH

end_time="$(timestamp)"
printf "%s,%s\n" "${start_time}" "${end_time}" >> "${EXP_DIR}/rewrites_selection_start_end_time.txt"

start_time="$(timestamp)"

EXEC="$ACCELOPT_BASE_DIR/scripts/fib/select_candidates.py"
EXECUTOR_RESULTS_PATH="$EXP_DIR/candidates/last_iteration_executor_results.json"
OUTPUT_BASE_PATH="$EXP_DIR/candidates"
python $EXEC --executor_results_path $EXECUTOR_RESULTS_PATH --output_base_path $OUTPUT_BASE_PATH --topk $TOPK_CANDIDATES

end_time="$(timestamp)"
printf "%s,%s\n" "${start_time}" "${end_time}" >> "${EXP_DIR}/select_candidates_start_end_time.txt"

cd $EXP_BASE_DIR