# MLflow + LocalStack ML Project

Production-like ML training and inference pipeline using MLflow, LocalStack S3, Postgres, and Docker Compose.

## Architecture

- **MLflow Server**: Experiment tracking and model registry
  - Backend: PostgreSQL
  - Artifact Store: LocalStack S3
- **Training**: PyTorch ResNet18 image classifier
- **Inference**: FastAPI server with model loading from MLflow
- **Infrastructure**: Fully containerized with Docker Compose

## Quick Start

### 1. Setup

```bash
# Clone and setup data directories
make setup

# Add some images to the created directories:
# - data/train/cat/
# - data/train/dog/
# - data/val/cat/
# - data/val/dog/
```

### 2. Start Services

```bash
# Start all services
make up

# Wait ~30 seconds for services to initialize
# Check MLflow UI: http://localhost:5000
```

### 3. Train Model

```bash
# Run training (will create dummy data if no images provided)
make train

# Training will:
# - Fine-tune ResNet18 on your data
# - Log metrics to MLflow
# - Register model as 'ivis_resnet'
# - Transition to Production stage
```

### 4. Test Inference

```bash
# Check health
curl http://localhost:8080/health

# Make prediction
curl -X POST -F "file=@your_image.jpg" http://localhost:8080/predict
```

## URLs

- MLflow UI: http://localhost:5000
- Inference API: http://localhost:8080
- API Docs: http://localhost:8080/docs

## Environment Variables

### MLflow Server
- `MLFLOW_BACKEND_STORE_URI`: PostgreSQL connection
- `MLFLOW_DEFAULT_ARTIFACT_ROOT`: S3 bucket path
- `MLFLOW_S3_ENDPOINT_URL`: LocalStack endpoint

### Trainer
- `DATA_ROOT`: Data directory (default: /data)
- `EXPERIMENT`: MLflow experiment name
- `MODEL_NAME`: Registered model name
- `EPOCHS`: Training epochs (default: 3)
- `BATCH`: Batch size (default: 32)
- `IMG`: Image size (default: 224)

### Inference
- `MLFLOW_TRACKING_URI`: MLflow server URL
- `MODEL_NAME`: Model to load from registry

## Commands

```bash
make help          # Show all commands
make up            # Start services
make down          # Stop services
make train         # Run training
make logs          # View logs
make clean         # Remove containers and volumes
```

## Troubleshooting

### LocalStack not ready
If you see S3 errors, wait 30 seconds for LocalStack to initialize:
```bash
docker logs ivis-localstack
```

### Missing data
The trainer will create dummy directories but needs actual images to train properly. Add .jpg files to:
- `data/train/<class>/`
- `data/val/<class>/`

### Model not in Production
The trainer automatically transitions the model to Production. If missing:
```bash
docker logs ivis-trainer
```

### Port conflicts
Default ports: MLflow (5000), Inference (8080), Postgres (5432), LocalStack (4566)

## Development

### Adding new dependencies
Update the relevant Dockerfile in `docker/` and rebuild:
```bash
make build
```

### Debugging
```bash
# Interactive shell in trainer
docker compose -f compose.local.yml run --rm trainer bash

# View MLflow logs
docker logs ivis-mlflow -f
```
