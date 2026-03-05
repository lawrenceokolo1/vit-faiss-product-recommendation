.PHONY: install download-abo build-index evaluate start-api start-mlflow docker-up test lint clean

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

download-abo:
	mkdir -p data/raw/abo/metadata data/raw/abo/images/metadata
	aws s3 sync s3://amazon-berkeley-objects/listings/metadata/ data/raw/abo/metadata/ --no-sign-request
	aws s3 cp s3://amazon-berkeley-objects/images/metadata/images.csv.gz data/raw/abo/images/metadata/images.csv.gz --no-sign-request
	aws s3 sync s3://amazon-berkeley-objects/images/small/ data/raw/abo/images/ --no-sign-request
	aws s3 sync s3://amazon-berkeley-objects/abo-benchmark-retrieval/ data/raw/abo/benchmark/ --no-sign-request 2>/dev/null || true

build-index:
	python pipelines/build_index.py

build-index-smoke:
	python pipelines/build_index.py --limit 100 --no-mlflow

evaluate:
	python pipelines/evaluate.py

visualize:
	@echo "Usage: python scripts/visualize_recommendations.py <path/to/image.jpg> [--top-k 5] [--out report.png]"
	@echo "Ensure API is running: make start-api"

start-api:
	uvicorn api.main:app --reload --port 8000

start-mlflow:
	mlflow server --host 0.0.0.0 --port 5000

docker-up:
	docker-compose up --build

test:
	pytest tests/ -v --tb=short

lint:
	black src/ api/ pipelines/ tests/
	isort src/ api/ pipelines/ tests/

clean:
	rm -rf artifacts/ mlflow/
	rm -rf data/processed/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
