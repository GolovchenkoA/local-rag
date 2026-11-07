"""Command line: python -m local_rag <parse|chunk|embed> [--book ID|all] [--force]."""

from __future__ import annotations

import argparse
import logging

from . import pipeline
from .config import load_config, select_books

STAGES = {"parse": pipeline.parse, "chunk": pipeline.chunk, "embed": pipeline.embed}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="local_rag", description=__doc__)
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--book", default="all", help="book_id from config.yaml, or 'all'")
    ap.add_argument("--force", action="store_true", help="redo even when output exists")
    ap.add_argument("--config", default=None, help="path to config.yaml")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    for book in select_books(cfg, args.book):
        STAGES[args.stage](cfg, book, force=args.force)


if __name__ == "__main__":
    main()
