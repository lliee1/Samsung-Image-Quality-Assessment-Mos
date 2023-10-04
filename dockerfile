FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

USER root

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install python3-pip -y && \
    apt install python3.8 -y && apt-get -y install libgl1-mesa-glx

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ="Asia/Seoul"
RUN apt install git -y && apt-get install libglib2.0-0 -y


RUN pip install --upgrade pip


WORKDIR /root
COPY . /root/dacon
WORKDIR /root/dacon/lightning-hydra-template

# requirements
RUN pip install -r requirements.txt

# warmup
RUN pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'