"""Pipeline stages: parse, chunk, embed. Each works on one book and skips finished output."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

from . import docling_client
from .chunking import build_chunks, read_jsonl, write_jsonl
from .config import BookPaths

log = logging.getLogger(__name__)


def parse(cfg: SimpleNamespace, book: SimpleNamespace, force: bool = False) -> None:
    """PDF -> data/parsed/<book_id>/book.docling.json + book.md (needs the Docling container)."""
    p = BookPaths(cfg, book)
    if p.docling_json.exists() and p.markdown.exists() and not force:
        log.info("[%s] parse: output exists, skipped (use --force)", book.book_id)
        return
    if not p.pdf.exists():
        raise FileNotFoundError(f"PDF not found: {p.pdf}")
    result = docling_client.convert_pdf(cfg, p.pdf)
    doc = result["document"]
    p.parsed_dir.mkdir(parents=True, exist_ok=True)
    p.docling_json.write_text(json.dumps(doc["json_content"], ensure_ascii=False), encoding="utf-8")
    p.markdown.write_text(doc["md_content"], encoding="utf-8")
    log.info("[%s] parse: wrote %s and %s", book.book_id, p.docling_json.name, p.markdown.name)


def chunk(cfg: SimpleNamespace, book: SimpleNamespace, force: bool = False) -> None:
    """Docling chunker response -> data/chunks/<book_id>/chunks.jsonl.

    The raw server response is saved as data/parsed/<book_id>/chunks.raw.json. When it
    already exists the Docling container is not needed (post-processing only).
    """
    p = BookPaths(cfg, book)
    if p.chunks.exists() and not force:
        log.info("[%s] chunk: output exists, skipped (use --force)", book.book_id)
        return
    if p.chunks_raw.exists() and not force:
        raw = json.loads(p.chunks_raw.read_text(encoding="utf-8"))
        log.info("[%s] chunk: using saved %s", book.book_id, p.chunks_raw.name)
    else:
        if not p.pdf.exists():
            raise FileNotFoundError(f"PDF not found: {p.pdf}")
        raw = docling_client.chunk_pdf(cfg, p.pdf)
        p.parsed_dir.mkdir(parents=True, exist_ok=True)
        p.chunks_raw.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    records = build_chunks(raw, book, cfg.chunking.chunker_version,
                           cfg.chunking.skip_chunks_without_headings)
    write_jsonl(p.chunks, records)
    log.info("[%s] chunk: %d server chunks -> %d kept -> %s",
             book.book_id, len(raw["chunks"]), len(records), p.chunks)


def embed(cfg: SimpleNamespace, book: SimpleNamespace, force: bool = False) -> None:
    """chunks.jsonl -> data/embeddings/<book_id>/dense.npy + meta.json (Python only)."""
    from . import embedding

    p = BookPaths(cfg, book)
    if not p.chunks.exists():
        raise FileNotFoundError(f"{p.chunks} not found; run the chunk stage first")
    if not force and embedding.embeddings_are_current(
            p.chunks, p.vectors, p.embeddings_meta, cfg.embedding.model):
        log.info("[%s] embed: vectors are current, skipped (use --force)", book.book_id)
        return
    records = read_jsonl(p.chunks)
    model = embedding.load_model(cfg)
    vectors = embedding.embed_texts(model, cfg, [r["text"] for r in records])
    meta = embedding.write_embeddings(
        vectors, [r["id"] for r in records], p.chunks, p.vectors, p.embeddings_meta,
        cfg.embedding.model)
    log.info("[%s] embed: %d vectors, dim %d -> %s",
             book.book_id, meta["count"], meta["dim"], p.vectors)
