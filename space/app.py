"""
Visual Product Recommendation — Gradio Space.

Standard HF pattern:
  - HF Dataset repo  → precomputed FAISS index + metadata (upload once via upload_index_to_hf.py)
  - HF Model         → ViT (from_pretrained)
  - This Space       → loads both at startup, then: embed query → FAISS search → return results

No external API. Set HF_DATASET_REPO_ID in Space Settings if not using default.
"""

import os
from pathlib import Path

import gradio as gr

# In-Space state (loaded once at startup)
_index = None
_metadata = None
_processor = None
_model = None
_loaded_from = None

SPACE_ROOT = Path(__file__).resolve().parent
ASSETS = SPACE_ROOT / "assets"
REPO_ID_FILE = SPACE_ROOT / "hf_dataset_repo_id.txt"


def _read_repo_id():
    """Single source of truth: env override, else space/hf_dataset_repo_id.txt."""
    out = os.environ.get("HF_DATASET_REPO_ID", "").strip()
    if out:
        return out
    if REPO_ID_FILE.exists():
        return REPO_ID_FILE.read_text().strip()
    return ""


HF_DATASET_REPO_ID = _read_repo_id()
API_URL = os.environ.get("API_URL", "").rstrip("/")


def _load_from_hf_dataset():
    """Load index + metadata from HF Dataset repo (standard pattern)."""
    global _index, _metadata, _processor, _model, _loaded_from
    try:
        from huggingface_hub import snapshot_download
        import faiss
        import pandas as pd
        import torch
        from transformers import AutoImageProcessor, AutoModel

        print("Loading index from HuggingFace Dataset...")
        artifacts_path = snapshot_download(
            repo_id=HF_DATASET_REPO_ID,
            repo_type="dataset",
            cache_dir=os.environ.get("HF_HOME", "/tmp/hf"),
        )
        artifacts_path = Path(artifacts_path)

        _index = faiss.read_index(str(artifacts_path / "product_index.faiss"))
        _metadata = pd.read_parquet(artifacts_path / "product_metadata.parquet")
        print(f"Index loaded: {_index.ntotal} products")

        _processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        _model = AutoModel.from_pretrained("google/vit-base-patch16-224")
        _model.eval()
        _loaded_from = "hf_dataset"
        return True
    except Exception as e:
        print("HF Dataset load failed:", e)
        return False


def _load_from_assets():
    """Fallback: load from space/assets/ (files copied into Space repo)."""
    global _index, _metadata, _processor, _model, _loaded_from
    faiss_path = ASSETS / "product_index.faiss"
    meta_path = ASSETS / "product_metadata.parquet"
    if not faiss_path.exists() or not meta_path.exists():
        return False
    try:
        import faiss
        import pandas as pd
        import torch
        from transformers import AutoImageProcessor, AutoModel

        _index = faiss.read_index(str(faiss_path))
        _metadata = pd.read_parquet(meta_path)
        _processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        _model = AutoModel.from_pretrained("google/vit-base-patch16-224")
        _model.eval()
        _loaded_from = "assets"
        return True
    except Exception as e:
        print("Assets load failed:", e)
        return False


def _embed(image_path: str):
    """Embed one image with ViT; return L2-normalized 768-d vector."""
    import numpy as np
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    inputs = _processor(images=img, return_tensors="pt")
    with __import__("torch").no_grad():
        out = _model(**inputs)
    emb = out.last_hidden_state[:, 0, :].numpy().astype(np.float32)
    norm = np.linalg.norm(emb) + 1e-8
    emb = emb / norm
    return emb


def _recommend_in_space(image_path: str, top_k: int):
    """Run inference in-Space: embed → FAISS search → metadata lookup."""
    import numpy as np

    emb = _embed(image_path).reshape(1, -1)
    k = min(top_k, _index.ntotal)
    scores, indices = _index.search(emb, k)

    rows = []
    for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
        if 0 <= idx < len(_metadata):
            row = _metadata.iloc[idx]
            title = str(row.get("title", ""))[:50]
            brand = str(row.get("brand", ""))
            cat = str(row.get("category", ""))[:30]
            rows.append((title, brand, cat, float(score)))
        else:
            rows.append(("—", "", "", float(score)))

    text = f"**Source:** {_loaded_from} · **Index size:** {_index.ntotal:,}\n\n"
    text += "| # | Title | Brand | Category | Score |\n|--|-------|-------|----------|-------|\n"
    for i, r in enumerate(rows, 1):
        text += f"| {i} | {r[0]} | {r[1]} | {r[2]} | {r[3]:.3f} |\n"
    return text


def _recommend_via_api(image_path: str, top_k: int):
    """Call external API (e.g. Fly.io)."""
    import httpx
    with open(image_path, "rb") as f:
        files = {"image": (Path(image_path).name, f.read(), "image/jpeg")}
    r = httpx.post(f"{API_URL}/recommend", files=files, data={"top_k": top_k}, timeout=60.0)
    r.raise_for_status()
    out = r.json()
    results = out.get("results", [])
    text = f"**API** · Latency: {out.get('latency_ms', 0):.0f} ms · Index: {out.get('index_size', 0):,}\n\n"
    text += "| # | Title | Brand | Category | Score |\n|--|-------|-------|----------|-------|\n"
    for i, row in enumerate(results, 1):
        title = (row.get("title") or "")[:50]
        text += f"| {i} | {title} | {row.get('brand', '')} | {(row.get('category') or '')[:30]} | {row.get('similarity_score', 0):.3f} |\n"
    return text


def recommend(image, top_k: int = 5):
    if not image:
        return None, "Upload an image."

    if _index is not None and _metadata is not None:
        try:
            text = _recommend_in_space(image, top_k)
            return image, text
        except Exception as e:
            return None, f"In-Space error: {e}"

    if API_URL:
        try:
            text = _recommend_via_api(image, top_k)
            return image, text
        except Exception as e:
            return None, f"API error: {e}"

    return (
        None,
        "No index loaded. Either upload the index to an HF Dataset and set **HF_DATASET_REPO_ID** "
        "(e.g. `your-hf-username/abo-visual-search-index`), or set **API_URL** to your API."
    )


# Startup: load index from HF Dataset (if repo set) or from local assets
if HF_DATASET_REPO_ID:
    _load_from_hf_dataset()
if _index is None:
    _load_from_assets()

with gr.Blocks(title="Visual Product Recommender", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Visual Product Recommendation\nUpload a product image → find similar items (ViT + FAISS on ABO).")
    with gr.Row():
        inp = gr.Image(type="filepath", label="Product image")
        top_k = gr.Slider(1, 20, value=5, step=1, label="Top-k")
    btn = gr.Button("Find similar")
    with gr.Row():
        out_img = gr.Image(label="Query", interactive=False)
        out_text = gr.Markdown(label="Results")
    btn.click(fn=recommend, inputs=[inp, top_k], outputs=[out_img, out_text])
    gr.Markdown(
        "Index from [HuggingFace Dataset](https://huggingface.co/datasets/) · "
        "[Amazon Berkeley Objects](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)"
    )

demo.launch()
