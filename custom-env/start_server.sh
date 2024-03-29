#!/bin/sh
# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
echo "Starting Custom Model environment with Triton inference server"

if [ "${ENABLE_CUSTOM_MODEL_RUNTIME_ENV_DUMP}" = 1 ]; then
    echo "Environment variables:"
    env
fi


echo
echo "Starting flask in background..."
echo
export FLASK_RUN_PORT=8080
export FLASK_RUN_HOST="0.0.0.0"
export FLASK_APP=${CODE_DIR}/hello.py
export MODEL_DIR="${CODE_DIR}/model_repository/"

exec nohup flask run > log.txt 2>&1 &

echo
echo "Executing command: tritonserver --model-repository=${MODEL_DIR}"
echo
exec tritonserver --model-repository=${MODEL_DIR}
