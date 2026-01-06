#!/bin/bash
EXEC="test_signoz.py"
export OTEL_SERVICE_NAME="test_signoz"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"
export OTEL_EXPORTER_OTLP_PROTOCOL="http/protobuf"
export LOGFIRE_SEND_TO_LOGFIRE="false"
export LOGFIRE_ENVIRONMENT=$OTEL_SERVICE_NAME
python $EXEC