FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

WORKDIR /app

RUN mkdir -p /results /path

RUN apt-get update && apt-get install -y \
    curl \
    git \
    python3.8 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o ~/miniconda.sh \
    && sh ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda update -n base -c defaults conda

RUN conda create -n di python=3.8 \
    && conda activate di \
    && conda install  pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia \
    && conda install pillow tensorboard\
    && conda clean -ya

RUN echo "conda activate di" >> ~/.bashrc

COPY utils/* ./
COPY models/* ./ 
COPY deep_inversion.py ./ 
COPY main.py ./
COPY train_with_distilled.py ./
