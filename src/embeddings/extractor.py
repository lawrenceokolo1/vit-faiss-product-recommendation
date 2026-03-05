"""
ViT image embedding extractor.

Uses google/vit-base-patch16-224 to produce 768-d L2-normalized embeddings.
"""

from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from src.utils.config import EMBED_BATCH_SIZE, VIT_MODEL_NAME


class ViTExtractor:
    """Extract 768-d image embeddings using ViT; outputs L2-normalized vectors for cosine similarity."""

    def __init__(self, model_name: str = VIT_MODEL_NAME, device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _load_image(self, path: Union[str, Path]) -> Image.Image:
        with open(path, "rb") as f:
            return Image.open(f).convert("RGB")

    def encode_image(
        self, image: Union[str, Path, Image.Image, np.ndarray]
    ) -> np.ndarray:
        """Encode a single image to 768-d L2-normalized vector."""
        if isinstance(image, (str, Path)):
            image = self._load_image(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        # CLS token
        emb = out.last_hidden_state[:, 0].cpu().numpy().astype(np.float32).squeeze()
        return emb / (np.linalg.norm(emb) + 1e-8)

    def encode_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = EMBED_BATCH_SIZE,
    ) -> np.ndarray:
        """Encode a list of images in batches; returns (N, 768) L2-normalized."""
        loaded = []
        for im in images:
            if isinstance(im, (str, Path)):
                loaded.append(self._load_image(im))
            else:
                loaded.append(im)
        all_embs = []
        for i in range(0, len(loaded), batch_size):
            batch = loaded[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**inputs)
            embs = out.last_hidden_state[:, 0].cpu().numpy().astype(np.float32)
            for j in range(embs.shape[0]):
                e = embs[j] / (np.linalg.norm(embs[j]) + 1e-8)
                all_embs.append(e)
        return np.stack(all_embs, axis=0)
