"""
Full pipeline: load ABO listings → split → embed (ViT + text) → fuse → build FAISS index → save + optional MLflow log.
"""

import resource
import sys
import time
from pathlib import Path

# Ensure project root is on path so "src" resolves when run from any directory
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.data.loader import load_listings, load_listings_from_images, get_image_path, save_listings_parquet
from src.data.splitter import create_splits, load_split_ids
from src.embeddings.extractor import ViTExtractor
from src.embeddings.text_encoder import TextEncoder
from src.embeddings.fusion import fuse_batch
from src.embeddings.indexer import FAISSIndexer
from src.utils.config import (
    ABO_METADATA_DIR,
    ABO_IMAGES_DIR,
    LISTINGS_PARQUET_PATH,
    TRAIN_IDS_PATH,
    ARTIFACTS_DIR,
    INDEX_PATH,
    METADATA_PATH,
    VIT_MODEL_NAME,
    TEXT_MODEL_NAME,
    DEFAULT_FUSION_ALPHA,
    EMBEDDING_DIM,
    EMBED_BATCH_SIZE,
    RANDOM_SEED,
)

# Optional MLflow (Phase 4)
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def run(
    metadata_dir: Path | None = None,
    images_dir: Path | None = None,
    fusion_alpha: float = DEFAULT_FUSION_ALPHA,
    limit: int | None = None,
    use_mlflow: bool = True,
    seed: int = RANDOM_SEED,
    from_images_only: bool = False,
) -> dict:
    """
    Load listings (or from parquet if already processed), create splits,
    embed index set with ViT + text, fuse, build FAISS index, save to artifacts.
    If from_images_only=True, build listing table from images on disk (use when
    metadata main_image_id does not match image filenames, e.g. ABO small hex ids).
    """
    metadata_dir = metadata_dir or ABO_METADATA_DIR
    images_dir = images_dir or ABO_IMAGES_DIR

    if LISTINGS_PARQUET_PATH.exists() and not from_images_only:
        df = pd.read_parquet(LISTINGS_PARQUET_PATH)
    elif from_images_only:
        df = load_listings_from_images(images_dir)
        if df.empty:
            raise FileNotFoundError(f"No images under {images_dir}. Download ABO images first.")
        save_listings_parquet(df)
        df = pd.read_parquet(LISTINGS_PARQUET_PATH)
    else:
        df = load_listings(metadata_dir=metadata_dir)
        if df.empty:
            raise FileNotFoundError(f"No listings found under {metadata_dir}. Download ABO metadata first.")
        save_listings_parquet(df)
        df = pd.read_parquet(LISTINGS_PARQUET_PATH)

    if limit:
        df = df.head(limit)
        save_listings_parquet(df)  # so create_splits uses same set → no "not in index"

    create_splits(listings_path=LISTINGS_PARQUET_PATH, seed=seed)
    _, _, index_ids = load_split_ids()
    if not index_ids:
        raise ValueError("No index IDs from split. Check data/processed/.")

    # Restrict to products we have in our split
    index_df = df[df["item_id"].astype(str).isin(index_ids)].set_index("item_id").loc[index_ids].reset_index()
    id_list = index_df["item_id"].astype(str).tolist()

    extractor = ViTExtractor(model_name=VIT_MODEL_NAME)
    text_encoder = TextEncoder(model_name=TEXT_MODEL_NAME)

    image_paths = []
    texts = []
    for _, row in index_df.iterrows():
        img_path = get_image_path(
            str(row["item_id"]),
            str(row.get("main_image_id", "")),
            images_dir,
            image_path_rel=row.get("image_path") if "image_path" in row and pd.notna(row.get("image_path")) else None,
        )
        image_paths.append(img_path)
        t = f"{row.get('title', '')} {row.get('product_type', '')} {row.get('color', '')} {row.get('material', '')}"
        texts.append(t.strip() or "unknown")

    # Filter to existing images only
    valid_indices = [i for i, p in enumerate(image_paths) if p.exists()]
    if not valid_indices:
        raise FileNotFoundError("No product images found. Download ABO images or point images_dir to correct path.")
    id_list = [id_list[i] for i in valid_indices]
    index_df = index_df.iloc[valid_indices].reset_index(drop=True)
    image_paths = [image_paths[i] for i in valid_indices]
    texts = [texts[i] for i in valid_indices]

    t0 = time.perf_counter()
    image_embs = extractor.encode_batch(image_paths, batch_size=EMBED_BATCH_SIZE)
    text_embs = text_encoder.encode_batch(texts)
    fused = fuse_batch(image_embs, text_embs, alpha=fusion_alpha)
    build_time = time.perf_counter() - t0
    avg_embed_ms = (build_time / len(id_list)) * 1000

    indexer = FAISSIndexer(dim=EMBEDDING_DIM)
    indexer.build(fused, id_list)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    indexer.save(INDEX_PATH, ARTIFACTS_DIR / "product_index_ids.txt")
    index_df.to_parquet(METADATA_PATH, index=False)
    index_size_mb = INDEX_PATH.stat().st_size / (1024 * 1024) if INDEX_PATH.exists() else 0

    # Peak RAM (ru_maxrss: bytes on macOS, KB on Linux)
    peak_ram_mb = None
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        rss = ru.ru_maxrss
        peak_ram_mb = (rss / (1024 * 1024)) if rss > 2**20 else (rss / 1024)
        peak_ram_mb = round(peak_ram_mb, 2)
    except Exception:
        pass

    metrics = {
        "build_time_seconds": build_time,
        "avg_embedding_time_ms": avg_embed_ms,
        "index_size_mb": round(index_size_mb, 2),
        "num_products": len(id_list),
    }
    if peak_ram_mb is not None:
        metrics["peak_ram_mb"] = peak_ram_mb

    if use_mlflow and MLFLOW_AVAILABLE:
        mlflow.set_experiment("visual-product-recommender")
        with mlflow.start_run():
            mlflow.log_params({
                "model_name": VIT_MODEL_NAME,
                "text_model": TEXT_MODEL_NAME,
                "fusion_alpha": fusion_alpha,
                "embedding_dim": EMBEDDING_DIM,
                "index_type": "IndexFlatIP",
                "num_products": len(id_list),
                "dataset": "ABO-v1",
            })
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(INDEX_PATH))
            mlflow.log_artifact(str(METADATA_PATH))
            if (ARTIFACTS_DIR / "product_index_ids.txt").exists():
                mlflow.log_artifact(str(ARTIFACTS_DIR / "product_index_ids.txt"))

    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of products (e.g. 1000 for smoke test)")
    parser.add_argument("--fusion-alpha", type=float, default=DEFAULT_FUSION_ALPHA)
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--from-images-only", action="store_true", help="Build listing table from images on disk (use for ABO small when metadata IDs don't match image filenames)")
    args = parser.parse_args()
    out = run(limit=args.limit, fusion_alpha=args.fusion_alpha, use_mlflow=not args.no_mlflow, seed=args.seed, from_images_only=args.from_images_only)
    print("Build complete:", out)
