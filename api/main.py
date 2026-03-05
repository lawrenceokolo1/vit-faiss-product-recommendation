"""
FastAPI app: /health, /model-info, /recommend, /recommend/by-id, /recommend/multimodal, /categories.
Loads FAISS index and metadata from artifacts/ (or MLflow fallback) on startup.
"""

import os
from pathlib import Path

# Load .env so HF_TOKEN, MLFLOW_TRACKING_URI etc. are set before models load
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import recommend
from api.schemas.models import HealthResponse, ModelInfoResponse

# Paths
ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
INDEX_PATH = ARTIFACTS_DIR / "product_index.faiss"
METADATA_PATH = ARTIFACTS_DIR / "product_metadata.parquet"
INDEX_IDS_PATH = ARTIFACTS_DIR / "product_index_ids.txt"

app = FastAPI(title="Visual Product Recommendation API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# State
_extractor = None
_text_encoder = None
_indexer = None
_metadata_df = None
_fusion_alpha = 0.7
_model_version = "1.0"


def _load_model_from_artifacts():
    """Load index and metadata from artifacts/ (fallback when MLflow not used)."""
    global _extractor, _text_encoder, _indexer, _metadata_df, _fusion_alpha, _model_version
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        return False
    try:
        from src.embeddings.extractor import ViTExtractor
        from src.embeddings.text_encoder import TextEncoder
        from src.embeddings.indexer import FAISSIndexer
        import pandas as pd
        from src.utils.config import DEFAULT_FUSION_ALPHA, EMBEDDING_DIM

        _extractor = ViTExtractor()
        _text_encoder = TextEncoder()
        _indexer = FAISSIndexer(dim=EMBEDDING_DIM)
        _indexer.load(INDEX_PATH, INDEX_IDS_PATH if INDEX_IDS_PATH.exists() else None)
        _metadata_df = pd.read_parquet(METADATA_PATH)
        _fusion_alpha = DEFAULT_FUSION_ALPHA
        _model_version = "1.0"
        recommend.init_recommend(
            _extractor, _text_encoder, _indexer, _metadata_df,
            _fusion_alpha, _model_version,
        )
        return True
    except Exception as e:
        print(f"Load from artifacts failed: {e}")
        return False


def _load_model_from_mlflow():
    """Try to load Production model from MLflow."""
    global _extractor, _text_encoder, _indexer, _metadata_df, _fusion_alpha, _model_version
    try:
        import mlflow
        import pandas as pd
        from src.embeddings.extractor import ViTExtractor
        from src.embeddings.text_encoder import TextEncoder
        from src.embeddings.indexer import FAISSIndexer
        from src.utils.config import EMBEDDING_DIM, MLFLOW_TRACKING_URI

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI))
        client = mlflow.MlflowClient()
        model_uri = f"models:/visual-product-recommender/Production"
        run = client.get_model_version("visual-product-recommender", "Production")
        # Download artifacts to local and load
        path = mlflow.artifacts.download_artifacts(run.source)
        # Simplified: assume we use artifacts/ as fallback; MLflow deploy would copy artifacts
        return False
    except Exception as e:
        print(f"MLflow load failed: {e}")
        return False


@app.on_event("startup")
def startup():
    if _load_model_from_artifacts():
        print("Model loaded from artifacts/")
        return
    if _load_model_from_mlflow():
        print("Model loaded from MLflow")
        return
    print("No model loaded. Run build_index pipeline and ensure artifacts/ has product_index.faiss and product_metadata.parquet.")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=_indexer is not None,
        model_version=_model_version if _indexer else "",
        index_size=len(_indexer.id_list) if _indexer else 0,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    from src.utils.config import VIT_MODEL_NAME, TEXT_MODEL_NAME
    return ModelInfoResponse(
        model_version=_model_version,
        fusion_alpha=_fusion_alpha,
        index_size=len(_indexer.id_list) if _indexer else 0,
        vit_model=VIT_MODEL_NAME,
        text_model=TEXT_MODEL_NAME,
    )


@app.get("/categories")
def categories():
    if _metadata_df is None or "category" not in _metadata_df.columns:
        return []
    return sorted(_metadata_df["category"].dropna().astype(str).unique().tolist())


app.include_router(recommend.router)
