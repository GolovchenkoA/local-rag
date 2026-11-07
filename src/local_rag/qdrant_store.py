"""Create the collection and upload prepared chunks + vectors to Qdrant.

Used by notebooks/upload-embeddings.ipynb. Nothing here runs on import.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import numpy as np
from qdrant_client import QdrantClient, models

from .chunking import read_jsonl
from .config import BookPaths
from .embedding import file_sha256

log = logging.getLogger(__name__)


def connect(cfg: SimpleNamespace) -> QdrantClient:
    return QdrantClient(url=cfg.qdrant.url)


def ensure_collection(client: QdrantClient, cfg: SimpleNamespace, collection: str,
                      dim: int) -> None:
    """Create the collection (named vector, payload indexes) when it does not exist."""
    if client.collection_exists(collection):
        log.info("Collection %s already exists", collection)
        return
    distance = getattr(models.Distance, cfg.qdrant.distance.upper())
    client.create_collection(
        collection_name=collection,
        vectors_config={cfg.qdrant.vector_name: models.VectorParams(size=dim, distance=distance)},
    )
    # book_id: filter by book. position: fetch neighbours with a range filter.
    client.create_payload_index(collection, "book_id", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(collection, "position", models.PayloadSchemaType.INTEGER)
    log.info("Created collection %s (dim=%d, %s)", collection, dim, cfg.qdrant.distance)


def prepare_points(cfg: SimpleNamespace, book: SimpleNamespace) -> list[models.PointStruct]:
    """Read chunks.jsonl + vectors and build the points. Checks that they belong together."""
    p = BookPaths(cfg, book)
    chunks = read_jsonl(p.chunks)
    vectors = np.load(p.vectors)
    meta = json.loads(p.embeddings_meta.read_text(encoding="utf-8"))

    if meta["chunks_sha256"] != file_sha256(p.chunks):
        raise ValueError(f"{book.book_id}: chunks.jsonl changed after embedding; run embed --force")
    if len(chunks) != len(vectors) or [c["id"] for c in chunks] != meta["ids"]:
        raise ValueError(f"{book.book_id}: chunks and vectors do not match; run embed --force")

    points = []
    for chunk, vector in zip(chunks, vectors):
        payload = {k: v for k, v in chunk.items() if k != "id"}
        payload["embedding_model"] = meta["model"]
        points.append(models.PointStruct(
            id=chunk["id"], vector={cfg.qdrant.vector_name: vector.tolist()}, payload=payload))
    return points


def upload_book(client: QdrantClient, cfg: SimpleNamespace, book: SimpleNamespace,
                collection: str, replace: bool = True, dry_run: bool = True) -> dict:
    """Upload one book. dry_run=True (default) only checks and reports, writes nothing.

    replace=True deletes all existing points of this book_id first, so a changed
    chunking never leaves old points behind.
    """
    points = prepare_points(cfg, book)
    dim = len(points[0].vector[cfg.qdrant.vector_name])
    report = {"book_id": book.book_id, "collection": collection, "points": len(points),
              "dim": dim, "replace": replace, "dry_run": dry_run}
    if dry_run:
        return report

    ensure_collection(client, cfg, collection, dim)
    if replace:
        client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(filter=models.Filter(must=[
                models.FieldCondition(key="book_id", match=models.MatchValue(value=book.book_id))
            ])),
        )
    client.upload_points(collection_name=collection, points=points,
                         batch_size=cfg.qdrant.upload_batch_size, max_retries=3, wait=True)
    report["count_in_qdrant"] = client.count(
        collection, exact=True,
        count_filter=models.Filter(must=[models.FieldCondition(
            key="book_id", match=models.MatchValue(value=book.book_id))])).count
    return report
