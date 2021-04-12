FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
# FROM pytorch/pytorch

RUN apt-get update -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY config/requirements.txt /opt/config/requirements.txt
RUN pip install -r /opt/config/requirements.txt

COPY src /opt/src
WORKDIR /opt/src