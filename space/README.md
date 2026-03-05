---
title: Visual Product Recommender
emoji: 🛒
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
---

# Visual Product Recommendation

Upload a product image → find visually similar items (ViT + FAISS on Amazon Berkeley Objects).

**How it works (standard HF pattern):**
- **HF Dataset repo** holds the precomputed FAISS index + metadata (full 130k+ products). The Space loads it at startup via `snapshot_download`.
- **ViT** is loaded from HuggingFace (`google/vit-base-patch16-224`).
- Each query: embed image → FAISS search → return top-k from metadata. No subset; full index.

**Index repo (single source of truth):** put your HF dataset repo id in `hf_dataset_repo_id.txt` (one line, e.g. `username/abo-visual-search-index`). The Space and the upload script both read it. You can override with **Variables** → `HF_DATASET_REPO_ID`. Optional: set `API_URL` to use an external API instead.
