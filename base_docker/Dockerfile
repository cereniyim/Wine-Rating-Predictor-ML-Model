FROM python:3.7-slim
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONPATH=/opt/orchestrator:$PYTHONPATH

COPY requirements.txt /opt/base_docker/

RUN pip install -r /opt/base_docker/requirements.txt

WORKDIR /opt/base_docker/
