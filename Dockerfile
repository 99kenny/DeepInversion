FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

RUN apt-get update && apt-get install -y \
    curl \
    git \
    python3.8 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

