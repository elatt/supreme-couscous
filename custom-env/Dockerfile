# This is the default base image for use with user models and workflows.
FROM nvcr.io/nvidia/tritonserver:24.02-vllm-python-py3

ENV PIP_NO_CACHE_DIR=1
RUN python3 -m venv /opt/dr && /opt/dr/bin/pip install --upgrade pip
COPY dr_requirements.txt dr_requirements.txt

RUN /opt/dr/bin/pip3 install -r dr_requirements.txt

# Copy the drop-in environment code into the correct directory
# Code from the custom model tarball can overwrite the code here
ENV CODE_DIR=/opt/code ADDRESS=0.0.0.0:8080
WORKDIR ${CODE_DIR}
COPY ./*.sh ./*.py ${CODE_DIR}/

ENTRYPOINT ["/opt/code/start_server.sh"]
