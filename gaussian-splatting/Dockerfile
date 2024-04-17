FROM nvidia/cuda:12.0.0-devel-ubi8
# FROM continuumio/anaconda3:2024.02-1

WORKDIR /gaussian-splatting

RUN wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
RUN bash Anaconda3-2024.02-1-Linux-x86_64.sh


RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt


RUN conda env create --file environment.yml
RUN conda activate gaussian_splatting


CMD ["python3", "main.py"]
