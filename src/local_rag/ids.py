"""Deterministic chunk IDs: loading the same book twice updates points instead of duplicating."""

from __future__ import annotations

import uuid

# Fixed namespace for this project. Never change it: it would change every ID.
NAMESPACE = uuid.UUID("6f1c2b5e-3d4a-4e8b-9a77-5c0f1a2b3c4d")


def chunk_id(book_id: str, headings: list[str], position: int) -> str:
    """UUIDv5 from book_id + full heading path + position (Qdrant IDs must be UUID or int)."""
    key = f"{book_id}|{' > '.join(headings)}|{position}"
    return str(uuid.uuid5(NAMESPACE, key))
