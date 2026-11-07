"""Load config.yaml. Access values with attribute syntax: cfg.chunking.max_tokens."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def _to_ns(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_ns(v) for v in value]
    return value


def load_config(path: Path | str | None = None) -> SimpleNamespace:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = _to_ns(yaml.safe_load(f))
    cfg.project_root = PROJECT_ROOT
    cfg.data_dir = (PROJECT_ROOT / cfg.paths.data_dir).resolve()
    return cfg


def get_book(cfg: SimpleNamespace, book_id: str) -> SimpleNamespace:
    for book in cfg.books:
        if book.book_id == book_id:
            return book
    known = ", ".join(b.book_id for b in cfg.books)
    raise KeyError(f"Unknown book_id {book_id!r}. Configured books: {known}")


def select_books(cfg: SimpleNamespace, book_id: str | None) -> list[SimpleNamespace]:
    """One book by id, or every configured book when book_id is None or 'all'."""
    if book_id in (None, "all"):
        return list(cfg.books)
    return [get_book(cfg, book_id)]


class BookPaths:
    """All files of one book under the data folder."""

    def __init__(self, cfg: SimpleNamespace, book: SimpleNamespace):
        d = cfg.data_dir
        self.pdf = d / "pdfs" / book.file
        self.parsed_dir = d / "parsed" / book.book_id
        self.docling_json = self.parsed_dir / "book.docling.json"
        self.markdown = self.parsed_dir / "book.md"
        self.chunks_raw = self.parsed_dir / "chunks.raw.json"  # raw Docling chunker response
        self.chunks_dir = d / "chunks" / book.book_id
        self.chunks = self.chunks_dir / "chunks.jsonl"
        self.embeddings_dir = d / "embeddings" / book.book_id
        self.vectors = self.embeddings_dir / "dense.npy"
        self.embeddings_meta = self.embeddings_dir / "meta.json"
