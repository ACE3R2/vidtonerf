FROM nvidia/cuda:11.6.2-devel-ubuntu20.04
# FROM continuumio/anaconda3:2024.02-1


SHELL ["/bin/bash", "-c"]


ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip build-essential wget ninja-build unzip libgl-dev ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH="/opt/conda/bin:$PATH"

WORKDIR /gaussian-splatting
COPY ./ ./
COPY ./gaussian-splatting/environment.yml environment.yml

ENV TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"

RUN conda update -n base conda
RUN conda install -n base conda-libmamba-solver
RUN conda config --set solver libmamba
RUN conda env create -f environment.yml
RUN conda init bash


SHELL ["conda", "run", "-n", "gaussian_splatting", "/bin/bash", "-c"]
RUN conda install jupyter colmap
RUN conda remove ffmpeg -y


# RUN conda env create --file environment.yml
# RUN conda install pytorch==2.0.0 torchvision==0.15.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
# RUN conda activate gaussian_splatting

WORKDIR /

#CMD ["python3", "main.py"]

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "gaussian_splatting", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
