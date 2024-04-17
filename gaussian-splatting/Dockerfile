FROM nvidia/cuda:11.7.1-devel-ubuntu20.04
# FROM continuumio/anaconda3:2024.02-1

WORKDIR /gaussian-splatting

SHELL ["/bin/bash", "-c"]


RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update -y && \
    apt-get install python3.7 -y && \
    apt-get install python3-pip -y && \
    apt-get install wget -y


RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh

# Puts in the string to simulate manual entry of enters and yeses
# TODO: find less hacky way around this (-y doesn't work because it is not apt-get command)
RUN echo -e '\nyes\n\nyes\n' | bash Anaconda3-2024.02-1-Linux-x86_64.sh

ENV PATH="/root/anaconda3/bin:$PATH"

COPY . .
COPY ./gaussian-splatting/environment.yml environment.yml

RUN conda env create --file environment.yml
# RUN conda install pytorch==2.0.0 torchvision==0.15.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
RUN conda activate gaussian_splatting



CMD ["python3", "main.py"]
