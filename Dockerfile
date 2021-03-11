FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY config/requirements.txt /opt/config/requirements.txt
RUN pip install -r /opt/config/requirements.txt

COPY src /opt/src
WORKDIR /opt/src