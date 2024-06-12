FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

LABEL maintainer="csanadlevente.balogh@edu.bme.hu"
LABEL docker_image_name="Supervised learning prioritization"
LABEL description="This container is created to train Supervised Learning models that adapt sample prioritization methods from Reinforcement Learning methodology"

WORKDIR /

RUN DEBIAN_FRONTEND=noninteractive apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -qy \
    python3-pip \
    build-essential \
    autoconf \
    automake \
    sudo \
    vim \
    nano \
    git \
    curl \
    wget \
    tmux

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

WORKDIR /sl_prioritized_sampling

COPY requirements.txt ./

RUN pip3 install -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/"
ENV PYTHONPATH "${PYTHONPATH}:/sl_prioritized_sampling/"

RUN mkdir -p experiments/models