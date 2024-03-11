#!/bin/sh
# Copyright 2021 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
echo "Starting Custom Model environment with NIM"

if [ "${ENABLE_CUSTOM_MODEL_RUNTIME_ENV_DUMP}" = 1 ]; then
    echo "Environment variables:"
    env
fi


echo
echo "Running NVIDIA init scripts..."
echo
/opt/nvidia/nvidia_entrypoint.sh /bin/true

echo
echo "Starting Flask proxy in background..."
echo
export FLASK_RUN_PORT=8080
export FLASK_RUN_HOST="0.0.0.0"
export FLASK_APP=${CODE_DIR}/hello.py
export MODEL_DIR="${CODE_DIR}/model-store/"

source /home/nemo/dr/bin/activate
nohup flask run > log.txt 2>&1 &

echo
echo "Downloading model files..."
echo
export AWS_ACCESS_KEY_ID="$(echo $MLOPS_RUNTIME_PARAM_s3Credential | jq -r .payload.awsAccessKeyId)"
export AWS_SECRET_ACCESS_KEY="$(echo $MLOPS_RUNTIME_PARAM_s3Credential | jq -r .payload.awsSecretAccessKey)"
export AWS_SESSION_TOKEN="$(echo $MLOPS_RUNTIME_PARAM_s3Credential | jq -r .payload.awsSessionToken)"
if [[ "$AWS_SESSION_TOKEN" == "null" ]]; then
    unset AWS_SESSION_TOKEN
fi

src="$(echo $MLOPS_RUNTIME_PARAM_s3Url | jq -r .payload)"
aws s3 cp --recursive "$src" /model-store/
unset AWS_ACCESS_KEY_ID
unset AWS_SECRET_ACCESS_KEY
unset AWS_SESSION_TOKEN


deactivate
echo
echo "Starting NeMo Inference Microservice..."
echo
# TODO: detect number of GPUs
exec nemollm_inference_ms --model llm --health_port=8081 --openai_port="9999" --nemo_port="9998" --num_gpus=1
