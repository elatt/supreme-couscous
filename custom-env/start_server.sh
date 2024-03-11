#!/bin/sh
# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
echo "Starting Custom Model environment with NIM"
set -e

if [ "${ENABLE_CUSTOM_MODEL_RUNTIME_ENV_DUMP}" = 1 ]; then
    echo "Environment variables:"
    env

    echo
    echo "Running NVIDIA init scripts..."
    echo
    /opt/nvidia/nvidia_entrypoint.sh /bin/true
fi

source /home/nemo/dr/bin/activate
echo
echo "Starting Flask proxy in background..."
echo
export FLASK_RUN_PORT=8080
export FLASK_RUN_HOST="0.0.0.0"
export FLASK_APP=${CODE_DIR}/hello.py
export MODEL_DIR="${CODE_DIR}/model-store/"

nohup flask run > log.txt 2>&1 &

if [[ -e $CODE_DIR/custom.py ]]; then
    echo
    echo "Running custom.py..."
    echo
    python ${CODE_DIR}/custom.py
fi
deactivate

echo
echo "Starting NeMo Inference Microservice..."
echo
exec nemollm_inference_ms --model llm \
    --health_port=8081 \
    --openai_port="9999" \
    --nemo_port="9998" \
    --num_gpus=$(nvidia-smi -L | wc -l)
