FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.3.1 \
    torchvision==0.18.1 \
    pillow==10.3.0 \
    mlflow==2.15.0 \
    boto3==1.28.57 \
    numpy==1.26.4 \
    tqdm==4.66.4

WORKDIR /app

COPY services/trainer/train.py /app/

