"""Central config for paths, model names, and constants."""

import os
from pathlib import Path

# Repo root (parent of src/)
ROOT_DIR = Path(__file__).resolve().parents[2]

# Data paths
DATA_RAW = ROOT_DIR / "data" / "raw" / "abo"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
ABO_METADATA_DIR = DATA_RAW / "metadata"
ABO_IMAGES_DIR = DATA_RAW / "images"
# ABO mapping: image_id -> path (same path as images/metadata on S3)
ABO_IMAGES_METADATA_DIR = ABO_IMAGES_DIR / "metadata"
ABO_IMAGES_CSV_PATH = ABO_IMAGES_METADATA_DIR / "images.csv.gz"
ABO_BENCHMARK_DIR = DATA_RAW / "benchmark"

# Artifacts (index + metadata written by pipeline)
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
INDEX_PATH = ARTIFACTS_DIR / "product_index.faiss"
METADATA_PATH = ARTIFACTS_DIR / "product_metadata.parquet"
EVAL_RESULTS_PATH = ARTIFACTS_DIR / "eval_results.json"

# Splits (written by splitter)
TRAIN_IDS_PATH = DATA_PROCESSED / "train_ids.txt"
QUERY_IDS_PATH = DATA_PROCESSED / "query_ids.txt"
INDEX_IDS_PATH = DATA_PROCESSED / "index_ids.txt"
LISTINGS_PARQUET_PATH = DATA_PROCESSED / "listings.parquet"

# Model names
VIT_MODEL_NAME = "google/vit-base-patch16-224"
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 768
TEXT_EMBEDDING_DIM = 384

# Fusion
DEFAULT_FUSION_ALPHA = 0.7

# Split / evaluation
RANDOM_SEED = 42
DEFAULT_QUERY_RATIO = 0.05  # ~5% of products as query set
DEFAULT_TOP_K = 10

# Batch size for embedding (lower = less RAM; ~1–2 GB typical with batch 32 for ViT + text encoder)
EMBED_BATCH_SIZE = 32

# MLflow (override via env)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_NAME = "visual-product-recommender"
