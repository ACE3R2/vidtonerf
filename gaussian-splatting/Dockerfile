FROM nvidia/cuda:12.1.0-devel-ubuntu20.04


WORKDIR /gaussian-splatting
SHELL ["/bin/bash", "-c"]


# ENV DEBIAN_FRONTEND=noninteractive
#RUN apt-get update && \
#    apt-get install -y python3.10 build-essential wget ninja-build unzip libgl-dev && \
#    apt-get clean && \
#    rm -rf /var/lib/apt/lists/*

RUN apt-get update
RUN export DEBIAN_FRONTEND=noninteractive && apt-get install -y python3.10 python3-pip


COPY ./gaussian-splatting/requirements.txt requirements.txt
COPY . .
RUN python3 -m pip install --upgrade pip
# RUN pip install torchvision
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install plyfile
RUN pip install tqdm
RUN pip install gaussian-splatting/submodules/simple-knn
RUN pip install gaussian-splatting/submodules/diff-gaussian-rasterization


CMD ["python3", "train.py", "-s", "test_images/"]
