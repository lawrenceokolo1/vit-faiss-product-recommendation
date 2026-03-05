#!/usr/bin/env python3
"""
Visualize recommendation results: show query image alongside top-k results.

Usage:
  # API must be running: make start-api
  python scripts/visualize_recommendations.py path/to/query.jpg
  python scripts/visualize_recommendations.py path/to/query.jpg --top-k 10 --api http://127.0.0.1:8000
  python scripts/visualize_recommendations.py path/to/query.jpg --out report.png

Shows: [Query image] and a row of result images with title + similarity score.
"""

import argparse
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.utils.config import ABO_IMAGES_DIR, ARTIFACTS_DIR, METADATA_PATH


def get_result_image_path(metadata_df: pd.DataFrame, item_id: str, images_dir: Path) -> Path | None:
    """Resolve product image path from metadata (image_path column) and images_dir."""
    row = metadata_df[metadata_df["item_id"].astype(str) == str(item_id)]
    if row.empty:
        return None
    row = row.iloc[0]
    rel = row.get("image_path")
    if pd.isna(rel) or not str(rel).strip():
        return None
    p = images_dir / str(rel).strip()
    if p.exists():
        return p
    # Try under small/ in case layout differs
    p2 = images_dir / "small" / str(rel).strip()
    return p2 if p2.exists() else p


def main():
    parser = argparse.ArgumentParser(description="Visualize /recommend results: query image + top-k results.")
    parser.add_argument("image", type=Path, help="Path to query image (e.g. data/raw/abo/images/61/61cb7a56.jpg)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of recommendations to show (default 5)")
    parser.add_argument("--api", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--out", type=Path, default=None, help="Save figure to this path instead of displaying")
    parser.add_argument("--images-dir", type=Path, default=None, help="Base dir for product images (default: data/raw/abo/images)")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Error: image not found: {args.image}")
        sys.exit(1)

    # Call API
    try:
        import httpx
    except ImportError:
        print("Install httpx: pip install httpx")
        sys.exit(1)

    with open(args.image, "rb") as f:
        files = {"image": (args.image.name, f.read(), "image/jpeg")}
    data = {"top_k": args.top_k}

    try:
        r = httpx.post(f"{args.api.rstrip('/')}/recommend", files=files, data=data, timeout=60.0)
        r.raise_for_status()
        resp = r.json()
    except Exception as e:
        print(f"API request failed: {e}")
        print("Is the API running? Try: make start-api")
        sys.exit(1)

    results = resp.get("results", [])
    if not results:
        print("No results returned.")
        sys.exit(0)

    # Load metadata to resolve image paths
    images_dir = args.images_dir or ABO_IMAGES_DIR
    if not METADATA_PATH.exists():
        print(f"Metadata not found: {METADATA_PATH}. Run build_index first.")
        sys.exit(1)
    meta = pd.read_parquet(METADATA_PATH)

    # Collect image paths for query and results
    query_path = Path(args.image)
    result_paths = []
    titles = []
    scores = []
    for res in results:
        item_id = res.get("item_id", "")
        title = (res.get("title") or "")[:40] + ("..." if len(res.get("title") or "") > 40 else "")
        score = res.get("similarity_score", 0)
        p = get_result_image_path(meta, item_id, images_dir)
        result_paths.append(p)
        titles.append(title or item_id)
        scores.append(score)

    # Plot
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        print("Install matplotlib and Pillow: pip install matplotlib Pillow")
        sys.exit(1)

    n_results = len(result_paths)
    n_cols = max(n_results, 1)
    fig, axes = plt.subplots(2, n_cols, figsize=(2 * n_cols, 4))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    # Row 0: query in first cell only
    ax0 = axes[0, 0]
    ax0.imshow(Image.open(query_path).convert("RGB"))
    ax0.set_title("Query", fontsize=10)
    ax0.axis("off")
    for j in range(1, n_cols):
        axes[0, j].axis("off")
    # Row 1: results
    result_axes = axes[1, :] if n_cols > 1 else [axes[1, 0]]
    for ax, path, title, score in zip(result_axes, result_paths, titles, scores):
        if path and path.exists():
            ax.imshow(Image.open(path).convert("RGB"))
        else:
            ax.text(0.5, 0.5, "No image", ha="center", va="center")
        ax.set_title(f"{title}\nscore: {score:.3f}", fontsize=8)
        ax.axis("off")

    plt.suptitle("Recommendation results (query → top-k similar products)", fontsize=11)
    plt.tight_layout()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=120, bbox_inches="tight")
        print(f"Saved: {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
