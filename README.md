# Visual Product Recommendation Engine

Production-grade multi-category visual product recommendation using **ViT** + **FAISS** on the **Amazon Berkeley Objects (ABO)** dataset. Image + text late fusion; eProduct-style evaluation (Recall@K, mAP).

## Quick start

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

1. **Download ABO data** (metadata + **image mapping** + images):
   ```bash
   make download-abo
   ```
   The mapping file `images.csv.gz` (from `images/metadata/`) links listing `main_image_id` to image paths; without it, use `--from-images-only`.
2. **Build index** (use `--limit 100` for smoke test without full dataset):
   ```bash
   python pipelines/build_index.py --limit 100 --no-mlflow
   ```
3. **Start API**:
   ```bash
   make start-api
   ```
   Then: `GET http://localhost:8000/health`, `POST http://localhost:8000/recommend` with form `image` + `top_k`.

3. **Visualize results** (optional — check that recommendations look right):
   ```bash
   # With API running in another terminal
   python scripts/visualize_recommendations.py data/raw/abo/images/61/61cb7a56.jpg --top-k 5
   # Save to file instead of opening a window
   python scripts/visualize_recommendations.py data/raw/abo/images/61/61cb7a56.jpg --out report.png
   ```
   Shows the query image and the top-k result images with titles and similarity scores.

## Project structure

- `src/data/` — ABO loader, train/query/index splitter
- `src/embeddings/` — ViT extractor, text encoder, late fusion, FAISS indexer
- `src/evaluation/` — Recall@K, mAP, evaluator
- `api/` — FastAPI app, `/recommend`, `/health`, `/model-info`, `/categories`
- `pipelines/` — `build_index.py`, `evaluate.py`
- `tests/` — unit tests for extractor, fusion, indexer, metrics, API

## Design decisions

- **ViT:** Strong off-the-shelf vision encoder; no training.
- **FAISS:** Industry-standard, exact search at 148k scale (IndexFlatIP).
- **Late fusion:** Simple, interpretable combination of image + text embeddings; alpha tunable via MLflow.

## Process and explanation

See **[docs/PROCESS.md](docs/PROCESS.md)** for a full walkthrough: data flow, build steps, what every console message means (ViT load report, Build complete metrics), and how to explain the system in interviews or portfolio reviews.

## License and citation

Dataset: **Amazon Berkeley Objects (ABO)** — CC BY-NC 4.0.  
Models: ViT and sentence-transformers — see Hugging Face model cards.
