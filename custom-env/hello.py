import asyncio
import csv
import io
import json
import os
import sys

import numpy as np
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import *

from flask import Flask, request

app = Flask(__name__)
app.logger.setLevel("INFO")

url_prefix = os.environ.get("URL_PREFIX", "")
triton_server_port = os.environ.get("TRITON_PORT", "8000")

default_system_prompt = "You are a helpful AI assistant. Keep short answers of no more than 2 sentences."


@app.route(f"{url_prefix}/")
def health():
    return {'message': 'OK'}


@app.route(f"{url_prefix}/health/")
def lrs_health():
    return {'message': 'OK'}


@app.post(f"{url_prefix}/predictUnstructured/")
def predict_unstructured():
    # proxy request to Triton server
    request_json = request.json
    user_prompt = request_json["prompt"]
    system_prompt = request_json.get("system", default_system_prompt)
    result = asyncio.run(main(user_prompt, system_prompt))
    model_response_bytes = result['0'][0]
    return model_response_bytes.decode('utf-8') + "\n\n", 200


def _delete_instruction(text):
    model_instructions_marker = '[/INST]'
    marker_len = len(model_instructions_marker)
    return text[text.find(model_instructions_marker) + marker_len + 1:]


@app.post(f"{url_prefix}/predict/")
def predict_text_gen():
    app.logger.info("Headers: %s", request.headers)
    filestorage = request.files.get("X")  # prediction server / drum magic
    reader = csv.DictReader(io.TextIOWrapper(filestorage))
    predictions = []
    for i, row in enumerate(reader):
        app.logger.info("Row %d: %s", i, row)
        user_prompt = row["promptText"]
        system_prompt = row.get("system", default_system_prompt)
        result = asyncio.run(main(user_prompt, system_prompt))
        app.logger.info("results: %s", result)
        predictions.append(_delete_instruction(result['0'][0].decode('utf-8')))
    return {"predictions": predictions}, 200


def create_request(prompt, stream, request_id, sampling_parameters, model_name, send_parameters_as_tensor=True):
    inputs = []
    prompt_data = np.array([prompt.encode("utf-8")], dtype=np.object_)
    try:
        inputs.append(grpcclient.InferInput("PROMPT", [1], "BYTES"))
        inputs[-1].set_data_from_numpy(prompt_data)
    except Exception as e:
        print(f"Encountered an error {e}")

    stream_data = np.array([stream], dtype=bool)
    inputs.append(grpcclient.InferInput("STREAM", [1], "BOOL"))
    inputs[-1].set_data_from_numpy(stream_data)

    # Request parameters are not yet supported via BLS. Provide an
    # optional mechanism to send serialized parameters as an input
    # tensor until support is added
    if send_parameters_as_tensor:
        sampling_parameters_data = np.array(
            [json.dumps(sampling_parameters).encode("utf-8")], dtype=np.object_
        )
        inputs.append(grpcclient.InferInput("SAMPLING_PARAMETERS", [1], "BYTES"))
        inputs[-1].set_data_from_numpy(sampling_parameters_data)

    # Add requested outputs
    outputs = []
    outputs.append(grpcclient.InferRequestedOutput("TEXT"))

    # Issue the asynchronous sequence inference.
    return {
        "model_name": model_name,
        "inputs": inputs,
        "outputs": outputs,
        "request_id": str(request_id),
        "parameters": sampling_parameters
    }


async def main(user_prompt, sys_prompt):
    model_name = "vllm"
    sampling_parameters = {"temperature": "0.01", "top_p": "1.0", "top_k": 20, "max_tokens": 512}
    stream = False

    prompts = [user_prompt]
    results_dict = {}

    async with grpcclient.InferenceServerClient(
            url='localhost:8001', verbose=False
    ) as triton_client:
        # Request iterator that yields the next request
        async def async_request_iterator():
            try:
                for iter in range(1):
                    for i, prompt in enumerate(prompts):
                        prompt_id = 0 + (len(prompts) * iter) + i
                        results_dict[str(prompt_id)] = []
                        system_prompt = f"<<SYS>>\n{sys_prompt}\n<</SYS>>\n\n"
                        prompt = f"<s>[INST]{system_prompt}{prompt}[/INST]"
                        yield create_request(
                            prompt, stream, prompt_id, sampling_parameters, model_name
                        )
            except Exception as error:
                print(f"caught error in request iterator:  {error}")

        try:
            # Start streaming
            response_iterator = triton_client.stream_infer(
                inputs_iterator=async_request_iterator(),
                stream_timeout=None,
            )
            # Read response from the stream
            async for response in response_iterator:
                result, error = response
                if error:
                    print(f"Encountered error while processing: {error}")
                else:
                    output = result.as_numpy("TEXT")
                    for i in output:
                        results_dict[result.get_response().id].append(i)

        except InferenceServerException as error:
            print(error)
            sys.exit(1)

    return results_dict
