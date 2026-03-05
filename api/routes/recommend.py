"""Recommendation endpoints: /recommend, /recommend/by-id, /recommend/multimodal."""

import time
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from api.schemas.models import ProductResult, RecommendResponse

router = APIRouter(prefix="/recommend", tags=["recommend"])

# Injected by main on startup
_extractor = None
_text_encoder = None
_indexer = None
_metadata_df = None
_fusion_alpha = 0.7
_model_version = "1.0"
_images_base_url = "/images"


def init_recommend(
    extractor,
    text_encoder,
    indexer,
    metadata_df,
    fusion_alpha: float,
    model_version: str,
    images_base_url: str = "/images",
):
    global _extractor, _text_encoder, _indexer, _metadata_df, _fusion_alpha, _model_version, _images_base_url
    _extractor = extractor
    _text_encoder = text_encoder
    _indexer = indexer
    _metadata_df = metadata_df
    _fusion_alpha = fusion_alpha
    _model_version = model_version
    _images_base_url = images_base_url


def _row_to_result(row, score: float, image_id: str) -> ProductResult:
    return ProductResult(
        item_id=str(row.get("item_id", "")),
        title=str(row.get("title", "")),
        brand=str(row.get("brand", "")),
        category=str(row.get("category", "")),
        color=str(row.get("color", "")),
        similarity_score=float(score),
        image_url=f"{_images_base_url}/{image_id}.jpg" if image_id else "",
    )


@router.post("", response_model=RecommendResponse)
async def recommend(
    image: UploadFile = File(...),
    top_k: int = Form(10),
):
    """Upload image, get top-k visually similar products."""
    if _indexer is None or _metadata_df is None or _extractor is None:
        raise HTTPException(503, "Model not loaded")
    contents = await image.read()
    try:
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    t0 = time.perf_counter()
    img_emb = _extractor.encode_image(img)
    if _text_encoder and _fusion_alpha < 1.0:
        text_emb = _text_encoder.encode_text("")
        from src.embeddings.fusion import fuse_embeddings

        q_emb = fuse_embeddings(img_emb, text_emb, _fusion_alpha)
    else:
        q_emb = img_emb
    ids_scores = _indexer.search_ids(q_emb.reshape(1, -1), k=top_k)
    latency_ms = (time.perf_counter() - t0) * 1000

    ids, scores = ids_scores[0] if ids_scores else ([], [])
    results = []
    for i, (item_id, score) in enumerate(zip(ids, scores)):
        row = _metadata_df[_metadata_df["item_id"].astype(str) == str(item_id)]
        if row.empty:
            results.append(
                ProductResult(item_id=item_id, similarity_score=float(score))
            )
        else:
            row = row.iloc[0]
            results.append(
                _row_to_result(row.to_dict(), score, str(row.get("main_image_id", "")))
            )

    return RecommendResponse(
        query_id=str(uuid.uuid4()),
        model_version=_model_version,
        fusion_alpha=_fusion_alpha,
        latency_ms=round(latency_ms, 2),
        index_size=len(_indexer.id_list),
        results=results,
    )


@router.post("/by-id", response_model=RecommendResponse)
async def recommend_by_id(
    item_id: str = Form(...),
    top_k: int = Form(10),
):
    """Get top-k similar products by product ID (uses its image + text)."""
    if _indexer is None or _metadata_df is None or _extractor is None:
        raise HTTPException(503, "Model not loaded")
    row = _metadata_df[_metadata_df["item_id"].astype(str) == str(item_id)]
    if row.empty:
        raise HTTPException(404, f"Product not found: {item_id}")
    row = row.iloc[0]
    from src.data.loader import get_image_path
    from src.utils.config import ABO_IMAGES_DIR

    _img_rel = row.get("image_path")
    img_path = get_image_path(
        str(row["item_id"]),
        str(row.get("main_image_id", "")),
        ABO_IMAGES_DIR,
        image_path_rel=(
            str(_img_rel).strip() if _img_rel and str(_img_rel) != "nan" else None
        ),
    )
    if not img_path.exists():
        raise HTTPException(404, f"Image not found for product {item_id}")

    t0 = time.perf_counter()
    img_emb = _extractor.encode_image(str(img_path))
    if _text_encoder:
        text_emb = _text_encoder.encode_listing(
            str(row.get("title", "")),
            str(row.get("product_type", "")),
            str(row.get("color", "")),
            str(row.get("material", "")),
        )
        from src.embeddings.fusion import fuse_embeddings

        q_emb = fuse_embeddings(img_emb, text_emb, _fusion_alpha)
    else:
        q_emb = img_emb
    ids_scores = _indexer.search_ids(q_emb.reshape(1, -1), k=top_k + 1)
    latency_ms = (time.perf_counter() - t0) * 1000

    ids, scores = ids_scores[0] if ids_scores else ([], [])
    results = []
    for i, (pid, score) in enumerate(zip(ids, scores)):
        if str(pid) == str(item_id):
            continue
        r = _metadata_df[_metadata_df["item_id"].astype(str) == str(pid)]
        if r.empty:
            results.append(ProductResult(item_id=pid, similarity_score=float(score)))
        else:
            r = r.iloc[0]
            results.append(
                _row_to_result(r.to_dict(), score, str(r.get("main_image_id", "")))
            )
        if len(results) >= top_k:
            break

    return RecommendResponse(
        query_id=str(uuid.uuid4()),
        model_version=_model_version,
        fusion_alpha=_fusion_alpha,
        latency_ms=round(latency_ms, 2),
        index_size=len(_indexer.id_list),
        results=results,
    )


@router.post("/multimodal", response_model=RecommendResponse)
async def recommend_multimodal(
    text_query: str | None = Form(None),
    top_k: int = Form(10),
    image: UploadFile | None = File(None),
):
    """Query by image and/or text. If only text, use text embedding only."""
    if _indexer is None or _metadata_df is None:
        raise HTTPException(503, "Model not loaded")
    if not text_query and not image:
        raise HTTPException(400, "Provide at least text_query or image")
    t0 = time.perf_counter()
    q_emb = None
    if image and image.filename:
        contents = await image.read()
        import io

        import numpy as np
        from PIL import Image

        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_emb = _extractor.encode_image(img)
        if text_query and _text_encoder:
            text_emb = _text_encoder.encode_text(text_query)
            from src.embeddings.fusion import fuse_embeddings

            q_emb = fuse_embeddings(img_emb, text_emb, _fusion_alpha)
        else:
            q_emb = img_emb
    elif text_query and _text_encoder:
        q_emb = _text_encoder.encode_text(text_query)
    if q_emb is None:
        raise HTTPException(400, "Could not compute query embedding")
    ids_scores = _indexer.search_ids(q_emb.reshape(1, -1), k=top_k)
    latency_ms = (time.perf_counter() - t0) * 1000
    ids, scores = ids_scores[0] if ids_scores else ([], [])
    results = []
    for pid, score in zip(ids, scores):
        r = _metadata_df[_metadata_df["item_id"].astype(str) == str(pid)]
        if r.empty:
            results.append(ProductResult(item_id=pid, similarity_score=float(score)))
        else:
            r = r.iloc[0]
            results.append(
                _row_to_result(r.to_dict(), score, str(r.get("main_image_id", "")))
            )
    return RecommendResponse(
        query_id=str(uuid.uuid4()),
        model_version=_model_version,
        fusion_alpha=_fusion_alpha,
        latency_ms=round(latency_ms, 2),
        index_size=len(_indexer.id_list),
        results=results,
    )
