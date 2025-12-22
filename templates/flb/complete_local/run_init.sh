EXP_DATE=$1
LAST_EXP_DATE=$2
EXP_BASE_DIR=$3
PROJECT_NAME=$4
ORG_NAME=$5

EXP_DIR=$EXP_BASE_DIR/$EXP_DATE

cd $EXP_DIR

LAST_EXP_EXECUTOR_RESULTS_PATH="$EXP_BASE_DIR/$LAST_EXP_DATE/executor_results.json"

mkdir -p $EXP_DIR/candidates
cp $LAST_EXP_EXECUTOR_RESULTS_PATH $EXP_DIR/candidates/last_iteration_executor_results.json

LOGFIRE_ENV_NAME=$(echo $EXP_DATE | sed 's/-//g') # Remove "-" in the exp date
logfire projects use $PROJECT_NAME --org $ORG_NAME
echo $LOGFIRE_ENV_NAME > $EXP_DIR/logfire_env_name.txt
export LOGFIRE_ENVIRONMENT=$LOGFIRE_ENV_NAME
echo "LOGFIRE_ENVIRONMENT: $LOGFIRE_ENVIRONMENT"

cd $EXP_BASE_DIR