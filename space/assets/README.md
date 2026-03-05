# In-Space index (optional)

To run **fully on Hugging Face** (no external API):

1. **Build the index** (from repo root; use `--limit N` if you want a smaller bundle for this folder):
   ```bash
   python pipelines/build_index.py --no-mlflow
   ```

2. **Copy artifacts into this folder** (from repo root):
   ```bash
   cp artifacts/product_index.faiss artifacts/product_metadata.parquet artifacts/product_index_ids.txt space/assets/
   ```

3. **Commit and push** so the deploy workflow uploads the Space with these files. The Space will then run recommendations in-Space with no API_URL.

If this folder has no `.faiss`/`.parquet`/`.txt` files, the Space will ask for **API_URL** instead.
