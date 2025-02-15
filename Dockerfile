FROM tensorflow/tensorflow:latest-gpu

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git vim