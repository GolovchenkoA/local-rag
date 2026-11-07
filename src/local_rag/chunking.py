"""Turn the raw Docling hybrid-chunker response into chunks.jsonl records.

The server already enforces the token limit and puts the heading path at the top of
each chunk's text. Here we add the project fields and drop what must not be loaded.
Callout labels (TIP, NOTE, WARNING) are kept on purpose.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from .ids import chunk_id


def build_chunks(raw: dict, book: SimpleNamespace, chunker_version: str,
                 skip_chunks_without_headings: bool = True) -> list[dict]:
    skip_top = set(getattr(book, "skip_top_headings", []) or [])
    kept: list[dict] = []
    for c in raw["chunks"]:
        headings = list(c.get("headings") or [])
        if skip_chunks_without_headings and not headings:
            continue
        if headings and headings[0] in skip_top:
            continue
        kept.append({
            "book_id": book.book_id,
            "book_title": book.book_title,
            "chapter": headings[0] if headings else None,
            "headings": headings,
            "pages": sorted(set(c.get("page_numbers") or [])),
            "token_count": c.get("num_tokens"),
            "type": "chunk",
            "chunker_version": chunker_version,
            "text": c["text"],
        })
    # position = order in the book, contiguous after filtering (neighbours are position +/- 1)
    for position, rec in enumerate(kept):
        rec["position"] = position
        rec["id"] = chunk_id(book.book_id, rec["headings"], position)
    return [dict(id=r["id"], **{k: v for k, v in r.items() if k != "id"}) for r in kept]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
