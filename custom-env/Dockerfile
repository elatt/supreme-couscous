# This is the default base image for use with user models and workflows.
FROM nvcr.io/nvidia/tritonserver:24.02-vllm-python-py3

# Install the list of core requirements, e.g. sklearn, numpy, pandas, flask.
# **Don't modify this file!**
COPY dr_requirements.txt dr_requirements.txt

# '--upgrade-strategy eager' will upgrade installed dependencies
# according to package requirements or to the latest
RUN pip3 install -U --no-cache-dir pip && \
    pip3 install -U --upgrade-strategy eager --no-cache-dir -r dr_requirements.txt

# Copy the drop-in environment code into the correct directory
# Code from the custom model tarball can overwrite the code here
ENV HOME=/opt CODE_DIR=/opt/code ADDRESS=0.0.0.0:8080
WORKDIR ${CODE_DIR}
COPY ./*.sh ${CODE_DIR}/
COPY ./*.py ${CODE_DIR}/

ENV WITH_ERROR_SERVER=1
# Uncomment the following line to switch from Flask to uwsgi server
#ENV PRODUCTION=1 MAX_WORKERS=1 SHOW_STACKTRACE=1

ENTRYPOINT ["/opt/code/start_server.sh"]
