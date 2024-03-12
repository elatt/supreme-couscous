import csv
import http.client
import io
import os

from flask import Flask, request
from openai import OpenAI

app = Flask(__name__)
app.logger.setLevel("INFO")

url_prefix = os.environ.get("URL_PREFIX", "")
openapi_port = os.environ.get("OPENAI_PORT", "9999")
health_port = os.environ.get("HEALTH_PORT", "8081")
model_name = os.environ.get("MODEL_NAME", "gpt-3")

default_system_prompt = (
    "You are a helpful AI assistant. Keep short answers of no more than 2 sentences."
)

nim = OpenAI(base_url=f"http://localhost:{openapi_port}/v1", api_key="fake")


@app.route(f"{url_prefix}/")
@app.route(f"{url_prefix}/health/")
def health():
    conn = http.client.HTTPConnection(f"localhost:{health_port}")
    conn.request("GET", "/v1/health/ready")
    r1 = conn.getresponse()
    if r1.status == 200:
        return {"message": "OK"}
    return (
        "Triton server is not ready. Please check the logs for more information.",
        503,
    )


@app.post(f"{url_prefix}/predict/")
@app.post(f"{url_prefix}/predictions/")
@app.post(f"{url_prefix}/invocations/")
def predict():
    app.logger.debug("Headers: %s", request.headers)

    filestorage = request.files.get("X")  # prediction server / drum magic
    reader = csv.DictReader(io.TextIOWrapper(filestorage))
    predictions = []
    # TODO: I'm not sure if we are meant to treat each row as a separate request or
    # if it is supposed to represent a whole chat history. How we handle this loop
    # will depend on that; for now I'm assuming separate requests.
    for i, row in enumerate(reader):
        app.logger.debug("Row %d: %s", i, row)
        # TODO: use runtime param to get prompt field name like Buzok does
        user_prompt = row["promptText"]
        system_prompt = row.get("system", default_system_prompt)

        completions = nim.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            n=1,
            temperature=0.01,
            max_tokens=512,
        )
        app.logger.debug("results: %s", completions)
        predictions.append(completions.choices[0].message.content)
    return {"predictions": predictions}, 200
