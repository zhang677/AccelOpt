EXP_DATE=$1 
EXP_BASE_DIR=$2 
EXPERIENCE_LIST_PATH=$3
BREADTH=$4
NUM_SAMPLES=$5
EXP_N=$6
PROJECT_NAME=$7
ORG_NAME=$8
TRACESET_ROOT=$9

EXP_DIR=$EXP_BASE_DIR/$EXP_DATE
cd $EXP_DIR # Important because *.txt use relative paths

LOGFIRE_ENV_NAME=$(echo $EXP_DATE | sed 's/-//g') # Remove "-" in the exp date
logfire projects use $PROJECT_NAME --org $ORG_NAME
export LOGFIRE_ENVIRONMENT=$LOGFIRE_ENV_NAME
echo $LOGFIRE_ENV_NAME > $EXP_DIR/logfire_env_name.txt
echo "LOGFIRE_ENVIRONMENT: $LOGFIRE_ENVIRONMENT"

mkdir -p $EXP_DIR/rewrites
CONSTRUCT_EXPERIENCE_EXEC="$ACCELOPT_BASE_DIR/scripts/construct_experience.py"
CONSTRUCT_EXPERIENCE_OUTPUT_PATH="$EXP_DIR/rewrites/aggregated_rewrites_list.json"
python $CONSTRUCT_EXPERIENCE_EXEC \
        --is_first \
        --experience_list_path $EXPERIENCE_LIST_PATH \
        --output_path $CONSTRUCT_EXPERIENCE_OUTPUT_PATH \
        --n $EXP_N

mkdir -p $EXP_DIR/planner_prompts
PLANNER_PROMPT_CONSTRUCTOR_EXEC="$ACCELOPT_BASE_DIR/prompts/planner_prompts/construct_base_prompt.py"
PLANNER_PROMPT_CONSTRUCTOR_ORIGINAL_BASE_PROMPT_PATH="$ACCELOPT_BASE_DIR/prompts/flb/planner_prompts/base_prompt.txt"
PLANNER_PROMPT_CONSTRUCTOR_NEW_BASE_PROMPT_PATH="$EXP_DIR/planner_prompts/base_prompt.txt"
python $PLANNER_PROMPT_CONSTRUCTOR_EXEC --original_base_prompt_path $PLANNER_PROMPT_CONSTRUCTOR_ORIGINAL_BASE_PROMPT_PATH \
    --summarizer_output_list_path $CONSTRUCT_EXPERIENCE_OUTPUT_PATH \
    --new_base_prompt_path $PLANNER_PROMPT_CONSTRUCTOR_NEW_BASE_PROMPT_PATH

PLANNER_EXEC="$ACCELOPT_BASE_DIR/scripts/flb/planner.py"
PLANNER_OUTPUT_PATH="$EXP_DIR/planner_results.json"
PLANNER_USER_TEMPLATE_PATH="$ACCELOPT_BASE_DIR/prompts/flb/planner_prompts/planner_prompt_template.txt"
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

SINGLE_EXECUTOR_EXEC="$ACCELOPT_BASE_DIR/scripts/flb/executor_with_fixer.py"
EXECUTOR_BASE_PROMPT_PATH="$ACCELOPT_BASE_DIR/prompts/flb/executor_prompts/base_prompt.txt"
EXECUTOR_USER_TEMPLATE_PATH="$ACCELOPT_BASE_DIR/prompts/flb/executor_prompts/user_prompt_template.txt"
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
    --traceset_root $TRACESET_ROOT
#    --save_fields_path $SAVE_FIELDS_PATH

cd $EXP_BASE_DIR