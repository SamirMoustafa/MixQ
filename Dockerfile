# To build the image, run the following command:
# docker build -t mixq .
# To run the image, run the following command:
# docker run --gpus all --rm -ti --ipc=host --name mixq_instance mixq /bin/bash

# Base Image
ARG BASE_IMAGE=pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel
FROM ${BASE_IMAGE} as base

# Install common dependencies and utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        wget \
        sudo \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install the required Python packages
RUN python -m pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
COPY ./requirements.txt /workspace/requirements.txt
RUN python -m pip install -r ./requirements.txt

# Set the working directory
COPY ./ /workspace
WORKDIR /workspace
ENV PYTHONPATH /workspace:$PYTHONPATH
