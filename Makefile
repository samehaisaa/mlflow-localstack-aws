.PHONY: help up down build train inference-test logs clean setup

help:
	@echo "Available targets:"
	@echo "  setup          - Create data directories and sample data"
	@echo "  up             - Start all services"
	@echo "  down           - Stop all services"
	@echo "  build          - Build all Docker images"
	@echo "  train          - Run model training"
	@echo "  inference-test - Test inference API"
	@echo "  logs           - Show logs for all services"
	@echo "  clean          - Clean up containers and volumes"

setup:
	@echo "Creating data directories..."
	@mkdir -p data/train/cat data/train/dog data/val/cat data/val/dog
	@echo "Data directories created. Add some .jpg images to:"
	@echo "  - data/train/cat/"
	@echo "  - data/train/dog/"
	@echo "  - data/val/cat/"
	@echo "  - data/val/dog/"

up:
	docker compose -f compose.local.yml up -d --build

down:
	docker compose -f compose.local.yml down

build:
	docker compose -f compose.local.yml build

train:
	docker compose -f compose.local.yml run --rm trainer

inference-test:
	@echo "Testing health endpoint..."
	@curl -s http://localhost:8080/health | jq .
	@echo "\nTesting prediction endpoint (requires an image.jpg file)..."
	@if [ -f "image.jpg" ]; then \
		curl -s -X POST -F "file=@image.jpg" http://localhost:8080/predict | jq .; \
	else \
		echo "Please provide an image.jpg file for testing"; \
	fi

logs:
	docker compose -f compose.local.yml logs -f

clean:
	docker compose -f compose.local.yml down -v
	@echo "Containers and volumes removed."
