"""(Re)generate notebooks/upload-embeddings.ipynb. The notebook is committed without outputs.

    .venv\\Scripts\\python.exe scripts/make_upload_notebook.py
"""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "upload-embeddings.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


CELLS = [
    md("""
# Upload embeddings to Qdrant

Phase 2 of the pipeline. Only the **Qdrant** container is needed (`docker compose up -d qdrant`).
Phase 1 (`parse`, `chunk`, `embed`) must be finished: this notebook reads
`data/chunks/<book_id>/chunks.jsonl` and `data/embeddings/<book_id>/dense.npy`.

By default it uploads to the **test collection** (`qdrant.test_collection` in `config.yaml`).
Nothing is written to Qdrant until you set `DRY_RUN = False` in the upload cell.
"""),
    code("""
import logging

from local_rag.config import load_config, select_books
from local_rag import qdrant_store

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
cfg = load_config()

# Choose the target. Keep the test collection until the result is checked.
COLLECTION = cfg.qdrant.test_collection      # cfg.qdrant.collection = main collection
BOOK = "all"                                  # a book_id from config.yaml, or "all"
books = select_books(cfg, BOOK)
print("Target collection:", COLLECTION)
print("Books:", [b.book_id for b in books])
"""),
    md("## 1. Connect to Qdrant"),
    code("""
client = qdrant_store.connect(cfg)
print("Existing collections:", [c.name for c in client.get_collections().collections])
"""),
    md("""
## 2. Check the prepared files (nothing is written)

Verifies for every book that `chunks.jsonl` and the vectors belong together
(same file hash, same IDs, same count) and reports what would be uploaded.
"""),
    code("""
for book in books:
    print(qdrant_store.upload_book(client, cfg, book, COLLECTION, dry_run=True))
"""),
    md("""
## 3. Upload

`replace=True` first deletes all existing points of the book, so a changed chunking never leaves
old points behind. The upload is idempotent (stable point IDs).

Set `DRY_RUN = False` to write.
"""),
    code("""
DRY_RUN = True      # <-- set to False to upload

for book in books:
    print(qdrant_store.upload_book(client, cfg, book, COLLECTION, replace=True, dry_run=DRY_RUN))
"""),
    md("## 4. Check the result"),
    code("""
if client.collection_exists(COLLECTION):
    print("Points in", COLLECTION, ":", client.count(COLLECTION, exact=True).count)
    points, _ = client.scroll(COLLECTION, limit=2, with_payload=True, with_vectors=False)
    for p in points:
        print(p.id, p.payload["headings"], p.payload["pages"], p.payload["position"])
else:
    print("Collection", COLLECTION, "does not exist yet (nothing uploaded).")
"""),
]

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (local-rag)", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

for _i, _cell in enumerate(CELLS):
    _cell["id"] = f"cell-{_i:02d}"

if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(NOTEBOOK, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", OUT)
