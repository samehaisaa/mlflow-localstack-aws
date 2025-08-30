FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    mlflow==2.15.0 \
    psycopg2-binary==2.9.9 \
    boto3==1.28.57 \
    awscli-local==0.21

WORKDIR /app

EXPOSE 5000
