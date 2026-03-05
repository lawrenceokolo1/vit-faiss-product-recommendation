"""Pydantic request/response schemas for recommendation API."""

from typing import List

from pydantic import BaseModel, Field


class ProductResult(BaseModel):
    """One product in recommendation results."""

    item_id: str
    title: str = ""
    brand: str = ""
    category: str = ""
    color: str = ""
    similarity_score: float
    image_url: str = ""


class RecommendResponse(BaseModel):
    """Response for /recommend and /recommend/by-id."""

    query_id: str
    model_version: str = "1.0"
    fusion_alpha: float = 0.7
    latency_ms: float
    index_size: int
    results: List[ProductResult] = Field(default_factory=list)


class MultimodalRecommendRequest(BaseModel):
    """Body for /recommend/multimodal: optional image + text."""

    text_query: str | None = None
    top_k: int = 10


class HealthResponse(BaseModel):
    """Response for /health."""

    status: str = "ok"
    model_loaded: bool = False
    model_version: str = ""
    index_size: int = 0


class ModelInfoResponse(BaseModel):
    """Response for /model-info."""

    model_version: str = ""
    fusion_alpha: float = 0.0
    index_size: int = 0
    vit_model: str = ""
    text_model: str = ""
