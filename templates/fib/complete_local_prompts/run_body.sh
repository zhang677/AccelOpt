EXP_DATE=$1 
EXP_BASE_DIR=$2 
EXPERIENCE_LIST_PATH=$3
BREADTH=$4
NUM_SAMPLES=$5
EXP_N=$6
FIXER_STEPS=$7
TRACESET_ROOT=$8

EXP_DIR=$EXP_BASE_DIR/$EXP_DATE
cd $EXP_DIR

LOG_ENV_NAME=$(cat $EXP_DIR/log_env_name.txt)
export OTEL_SERVICE_NAME=$LOG_ENV_NAME 
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"
export OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf"
export LOGFIRE_SEND_TO_LOGFIRE="false"
export LOGFIRE_ENVIRONMENT=$LOG_ENV_NAME 
echo "OTEL_SERVICE_NAME: $OTEL_SERVICE_NAME"

CONSTRUCT_EXPERIENCE_EXEC="$ACCELOPT_BASE_DIR/scripts/construct_experience.py"
CONSTRUCT_EXPERIENCE_OUTPUT_PATH="$EXP_DIR/rewrites/aggregated_rewrites_list.json"
CONSTRUCT_EXPERIENCE_ORIGINAL_REWRITE_LIST_PATH="$EXP_DIR/rewrites/rewrites_selection_output_list.json"
python $CONSTRUCT_EXPERIENCE_EXEC --original_rewrite_list_path $CONSTRUCT_EXPERIENCE_ORIGINAL_REWRITE_LIST_PATH --experience_list_path $EXPERIENCE_LIST_PATH --output_path $CONSTRUCT_EXPERIENCE_OUTPUT_PATH --n $EXP_N

# Mkdir the planner_prompts directory if it doesn't exist
mkdir -p $EXP_DIR/planner_prompts
PLANNER_PROMPT_CONSTRUCTOR_EXEC="$ACCELOPT_BASE_DIR/prompts/planner_prompts/construct_base_prompt.py"
PLANNER_PROMPT_CONSTRUCTOR_ORIGINAL_BASE_PROMPT_PATH="$PROMPT_BASE_DIR/planner_prompts/base_prompt.txt"
PLANNER_PROMPT_CONSTRUCTOR_NEW_BASE_PROMPT_PATH="$EXP_DIR/planner_prompts/base_prompt.txt"
python $PLANNER_PROMPT_CONSTRUCTOR_EXEC --original_base_prompt_path $PLANNER_PROMPT_CONSTRUCTOR_ORIGINAL_BASE_PROMPT_PATH --summarizer_output_list_path $CONSTRUCT_EXPERIENCE_OUTPUT_PATH --new_base_prompt_path $PLANNER_PROMPT_CONSTRUCTOR_NEW_BASE_PROMPT_PATH

PLANNER_EXEC="$ACCELOPT_BASE_DIR/scripts/fib/planner.py"
PLANNER_OUTPUT_PATH="$EXP_DIR/planner_results.json"
PLANNER_USER_TEMPLATE_PATH="$PROMPT_BASE_DIR/planner_prompts/planner_prompt_template.txt"
PLANNER_PROFILE_RESULT_PATH="$EXP_DIR/candidates/profile_results.csv"
PLANNER_MODEL_CONFIG_PATH="$EXP_BASE_DIR/configs/planner_config.json"
# PLANNER_DISPLAYED_PROFILES_PATH="$ACCELOPT_BASE_DIR/prompts/planner_prompts/displayed_profiles.json"
python $PLANNER_EXEC --output_path $PLANNER_OUTPUT_PATH \
    --breadth $BREADTH \
    --exp_dir $EXP_DIR \
    --base_prompt_path $PLANNER_PROMPT_CONSTRUCTOR_NEW_BASE_PROMPT_PATH \
    --user_template_path $PLANNER_USER_TEMPLATE_PATH \
    --profile_result_path $PLANNER_PROFILE_RESULT_PATH \
    --model_config_path $PLANNER_MODEL_CONFIG_PATH
#    --displayed_profiles_path $PLANNER_DISPLAYED_PROFILES_PATH

SINGLE_EXECUTOR_EXEC="$ACCELOPT_BASE_DIR/scripts/fib/executor.py"
EXECUTOR_BASE_PROMPT_PATH="$PROMPT_BASE_DIR/executor_prompts/base_prompt.txt"
EXECUTOR_USER_TEMPLATE_PATH="$PROMPT_BASE_DIR/executor_prompts/user_prompt_template.txt"
FIXER_BASE_PROMPT_PATH="$PROMPT_BASE_DIR/fixer_prompts/base_prompt.txt"
FIXER_USER_TEMPLATE_PATH="$PROMPT_BASE_DIR/fixer_prompts/user_prompt_template.txt"
# SAVE_FIELDS_PATH="$ACCELOPT_BASE_DIR/prompts/profile_list.json"
EXECUTOR_MODEL_CONFIG_PATH="$EXP_BASE_DIR/configs/executor_config.json"
EXECUTOR_LOG_OUTPUT_PATH="$EXP_DIR/executor_results.json"
python $SINGLE_EXECUTOR_EXEC --num_samples $NUM_SAMPLES \
    --problems_path $PLANNER_PROFILE_RESULT_PATH \
    --extractor_output_path $PLANNER_OUTPUT_PATH \
    --exp_dir $EXP_DIR \
    --base_prompt_path $EXECUTOR_BASE_PROMPT_PATH \
    --user_template_path $EXECUTOR_USER_TEMPLATE_PATH \
    --model_config_path $EXECUTOR_MODEL_CONFIG_PATH \
    --output_path $EXECUTOR_LOG_OUTPUT_PATH \
    --exp_date $EXP_DATE \
    --traceset_root $TRACESET_ROOT \
    --fixer_steps $FIXER_STEPS \
    --fixer_base_prompt_path $FIXER_BASE_PROMPT_PATH \
    --fixer_user_template_path $FIXER_USER_TEMPLATE_PATH
#    --save_fields_path $SAVE_FIELDS_PATH \

cd $EXP_BASE_DIR