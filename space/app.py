"""
Gradio demo for Visual Product Recommendation.
Upload an image → get top-k similar products (from API or local index).
Set API_URL in Space settings to point to your running API for live recommendations.
"""

import os
from pathlib import Path

import gradio as gr

API_URL = os.environ.get("API_URL", "").rstrip("/")


def recommend(image, top_k: int = 5):
    if not image:
        return None, "Upload an image."
    if not API_URL:
        return (
            None,
            "No API URL configured. Set **API_URL** in this Space's Settings (or run the API locally). "
            "Then upload an image to get recommendations.",
        )
    import io
    import httpx
    with open(image, "rb") as f:
        files = {"image": (Path(image).name, f.read(), "image/jpeg")}
    data = {"top_k": top_k}
    try:
        r = httpx.post(f"{API_URL}/recommend", files=files, data=data, timeout=30.0)
        r.raise_for_status()
        out = r.json()
    except Exception as e:
        return None, f"API error: {e}"
    results = out.get("results", [])
    if not results:
        return None, "No results."
    text = f"**Model:** {out.get('model_version', '')}  \n**Latency:** {out.get('latency_ms', 0):.0f} ms  \n**Index size:** {out.get('index_size', 0):,}\n\n"
    text += "| # | Title | Brand | Category | Score |\n|--|-------|-------|----------|-------|\n"
    for i, row in enumerate(results, 1):
        title = (row.get("title") or "")[:50] + ("..." if len(row.get("title") or "") > 50 else "")
        brand = row.get("brand") or ""
        cat = (row.get("category") or "")[:30]
        score = row.get("similarity_score", 0)
        text += f"| {i} | {title} | {brand} | {cat} | {score:.3f} |\n"
    return image, text


with gr.Blocks(title="Visual Product Recommender", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Visual Product Recommendation\nUpload a product image to find similar items.")
    with gr.Row():
        inp = gr.Image(type="filepath", label="Product image")
        top_k = gr.Slider(1, 20, value=5, step=1, label="Top-k")
    btn = gr.Button("Get recommendations")
    with gr.Row():
        out_img = gr.Image(label="Query image", interactive=False)
        out_text = gr.Markdown(label="Results")
    btn.click(fn=recommend, inputs=[inp, top_k], outputs=[out_img, out_text])
    gr.Markdown(
        "Built with ViT + FAISS on [Amazon Berkeley Objects](https://amazon-berkeley-objects.s3.amazonaws.com/index.html). "
        "Set **API_URL** in Space settings to your running API for live results."
    )

demo.launch()
