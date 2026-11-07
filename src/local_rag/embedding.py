"""Dense embeddings with bge-m3 (sentence-transformers), CUDA first with CPU fallback."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .device import resolve_device

log = logging.getLogger(__name__)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_model(cfg: SimpleNamespace):
    from sentence_transformers import SentenceTransformer

    device = resolve_device(getattr(cfg.embedding, "device", None) or cfg.device)
    model = SentenceTransformer(cfg.embedding.model, device=device)
    model.max_seq_length = cfg.embedding.max_seq_length
    return model


def embed_texts(model, cfg: SimpleNamespace, texts: list[str]) -> np.ndarray:
    vectors = model.encode(
        texts,
        batch_size=cfg.embedding.batch_size,
        normalize_embeddings=cfg.embedding.normalize,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return vectors.astype(np.float32)


def write_embeddings(vectors: np.ndarray, ids: list[str], chunks_path: Path,
                     vectors_path: Path, meta_path: Path, model_name: str) -> dict:
    vectors_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path, vectors)
    meta = {
        "model": model_name,
        "dim": int(vectors.shape[1]),
        "count": int(vectors.shape[0]),
        "dtype": str(vectors.dtype),
        "chunks_sha256": file_sha256(chunks_path),
        "ids": ids,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(meta_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, ensure_ascii=False)
    return meta


def embeddings_are_current(chunks_path: Path, vectors_path: Path, meta_path: Path,
                           model_name: str) -> bool:
    """True when vectors exist for exactly this chunks file and this model."""
    if not (vectors_path.exists() and meta_path.exists() and chunks_path.exists()):
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta.get("model") == model_name and meta.get("chunks_sha256") == file_sha256(chunks_path)
