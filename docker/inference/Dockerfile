FROM python:3.11-slim

RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn==0.30.1 \
    python-multipart==0.0.9 \
    pillow==10.3.0 \
    mlflow==2.15.0 \
    boto3==1.28.57 \
    torch==2.3.1 \
    torchvision==0.18.1

WORKDIR /app

COPY services/inference/server.py /app/

EXPOSE 8080
