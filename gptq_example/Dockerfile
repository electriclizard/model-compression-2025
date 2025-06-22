FROM nvcr.io/nvidia/tritonserver:24.04-vllm-python-py3
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /src
COPY . $WORKDIR

RUN sh install.sh

ENTRYPOINT [ "/bin/bash" ]
