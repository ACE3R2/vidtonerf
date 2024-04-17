FROM python:3.10.8-slim


WORKDIR /gaussian-splatting
SHELL ["/bin/bash", "-c"]

COPY . .
COPY ./gaussian-splatting/requirements.txt requirements.txt

RUN python3.10 -m pip install --upgrade -r requirements.txt

CMD ["python3", "main.py"]
