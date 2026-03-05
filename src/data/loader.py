"""
ABO metadata parser and image path resolver.

Reads ABO listing JSONL from data/raw/abo/metadata/listings_*.json.gz (one JSON object per line),
joins with images/metadata/images.csv.gz (image_id -> path e.g. "1f/1f360d6f.jpg") so that
main_image_id (e.g. 81iZlv3bjpL) resolves to the actual image file under data/raw/abo/images/.
"""

import gzip
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.utils.config import (ABO_BENCHMARK_DIR, ABO_IMAGES_CSV_PATH,
                              ABO_IMAGES_DIR, ABO_METADATA_DIR, DATA_PROCESSED,
                              LISTINGS_PARQUET_PATH)


def _first_value(items: list[dict], key: str = "value") -> str:
    """Extract first value from ABO list-of-dicts (e.g. [{"value": "Nike"}])."""
    if not items or not isinstance(items, list):
        return ""
    first = items[0] if items else {}
    return first.get(key, "") or first.get("node_name", "") or ""


def _parse_listing(raw: dict) -> dict[str, Any]:
    """Convert one ABO listing JSON to flat dict. Prefer en_US title when available."""
    item_id = raw.get("item_id", "")
    item_name_list = raw.get("item_name", [])
    title = next(
        (
            t.get("value", "")
            for t in item_name_list
            if isinstance(t, dict) and t.get("language_tag") == "en_US"
        ),
        _first_value(item_name_list),
    )
    brand = _first_value(raw.get("brand", []))
    color = _first_value(raw.get("color", []))
    material = _first_value(raw.get("material", []))
    product_type = raw.get("product_type", "")
    if isinstance(product_type, list):
        product_type = _first_value(product_type)
    main_image_id = raw.get("main_image_id", "")
    node = raw.get("node", [])
    category = _first_value(node, "node_name") if node else ""

    return {
        "item_id": item_id,
        "title": title or "",
        "brand": brand,
        "color": color,
        "material": material,
        "product_type": product_type or "",
        "main_image_id": main_image_id,
        "category": category,
    }


def _load_abo_image_lookup(
    images_metadata_path: Optional[Path] = None,
) -> dict[str, str]:
    """
    Load ABO image_id -> path mapping from images.csv.gz (bridge between main_image_id and hex filenames).
    Prefers data/raw/abo/images/metadata/images.csv.gz; fallback: data/raw/abo/metadata/images.csv.gz.
    Returns dict: main_image_id -> relative path e.g. "1f/1f360d6f.jpg".
    """
    path = Path(images_metadata_path) if images_metadata_path else None
    if not path or not path.exists():
        path = ABO_IMAGES_CSV_PATH
    if not path.exists():
        path = ABO_METADATA_DIR / "images.csv.gz"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        if "image_id" not in df.columns or "path" not in df.columns:
            return {}
        return dict(zip(df["image_id"].astype(str), df["path"].astype(str)))
    except Exception:
        return {}


def load_listings(
    metadata_dir: Optional[Path] = None,
    use_benchmark: bool = False,
    require_image_path: bool = True,
) -> pd.DataFrame:
    """
    Load ABO listings from listings_*.json.gz and join with images.csv.gz so each row has
    image_path (relative path under images/). Returns columns: item_id, title, brand, color,
    material, product_type, main_image_id, category, image_path.
    If require_image_path=True (default), only rows with a matching image are kept.
    """
    base = ABO_BENCHMARK_DIR if use_benchmark else (metadata_dir or ABO_METADATA_DIR)
    if not base.exists():
        return pd.DataFrame()

    image_lookup = _load_abo_image_lookup(ABO_IMAGES_CSV_PATH)

    rows = []
    # ABO metadata: JSONL inside listings_*.json.gz only (exclude images.csv.gz)
    for path in sorted(base.glob("listings_*.json.gz")):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        rec = _parse_listing(item)
                        main_image_id = rec.get("main_image_id", "")
                        if main_image_id and image_lookup:
                            rel_path = image_lookup.get(main_image_id)
                            if rel_path:
                                rec["image_path"] = rel_path
                            elif not require_image_path:
                                rec["image_path"] = ""
                        elif not require_image_path:
                            rec["image_path"] = ""
                        if require_image_path and not rec.get("image_path"):
                            continue
                        rows.append(rec)
                    except json.JSONDecodeError:
                        continue
        except (OSError, TypeError):
            continue

    if not rows:
        return pd.DataFrame()

    df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["item_id"], keep="first")
        .reset_index(drop=True)
    )
    if "image_path" not in df.columns:
        df["image_path"] = ""
    return df


def get_image_path(
    item_id: str,
    main_image_id: str,
    images_dir: Optional[Path] = None,
    image_path_rel: Optional[str] = None,
) -> Path:
    """
    Resolve image path for a listing. If image_path_rel is provided (from ABO images.csv.gz join,
    e.g. "1f/1f360d6f.jpg"), use images_dir / image_path_rel. Otherwise fall back to main_image_id
    as filename (hex layout: images/<xx>/<id>.jpg).
    """
    base = Path(images_dir) if images_dir else ABO_IMAGES_DIR
    if image_path_rel and str(image_path_rel).strip():
        p = base / image_path_rel.strip()
        if p.exists():
            return p
        return p  # caller may still use for writing
    mid = str(main_image_id).strip()
    if not mid:
        return base / "unknown.jpg"
    if len(mid) >= 2:
        p = base / mid[:2] / f"{mid}.jpg"
        if p.exists():
            return p
    for ext in (".jpg", ".jpeg", ".png"):
        p = base / mid[:2] / f"{mid}{ext}" if len(mid) >= 2 else base / f"{mid}{ext}"
        if p.exists():
            return p
    return base / mid[:2] / f"{mid}.jpg" if len(mid) >= 2 else base / f"{mid}.jpg"


def get_listing_by_id(listings: pd.DataFrame, item_id: str) -> Optional[dict]:
    """Get a single listing as dict by item_id."""
    row = listings[listings["item_id"] == item_id]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def save_listings_parquet(df: pd.DataFrame, path: Optional[Path] = None) -> Path:
    """Save cleaned listings to parquet for fast reload."""
    path = path or LISTINGS_PARQUET_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_listings_from_images(
    images_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build a listing table from the images on disk (images/<xx>/<id>.jpg).
    Use when metadata main_image_id does not match image filenames (e.g. ABO small uses hex ids).
    Each row: item_id = main_image_id = image stem (e.g. 1f360d6f), title/brand/color empty.
    """
    base = Path(images_dir) if images_dir else ABO_IMAGES_DIR
    if not base.exists():
        return pd.DataFrame()
    rows = []
    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir():
            continue
        for f in subdir.glob("*.jpg"):
            stem = f.stem
            rows.append(
                {
                    "item_id": stem,
                    "title": "",
                    "brand": "",
                    "color": "",
                    "material": "",
                    "product_type": "",
                    "main_image_id": stem,
                    "category": "",
                }
            )
    return pd.DataFrame(rows)
