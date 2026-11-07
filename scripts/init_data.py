"""Create the gitignored ``data/`` folder structure if it is absent.

Safe to run many times: existing folders and files are left untouched.
Uses only the standard library, so it works with any Python 3 (no virtual
environment needed). Run it from anywhere:

    py scripts/init_data.py
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Top-level folders. Per-book folders (parsed/<book_id>/, chunks/<book_id>/,
# embeddings/<book_id>/) are created by the pipeline when a stage writes output.
DATA_FOLDERS = [
    "data/pdfs",         # source PDFs
    "data/parsed",       # Docling output per book
    "data/parsed/test",  # scratch folder for manual Docling API tests
    "data/chunks",       # chunks.jsonl per book
    "data/embeddings",   # vectors prepared for upload
    "data/discussions",  # local design notes
]


def main() -> None:
    for folder in DATA_FOLDERS:
        path = PROJECT_ROOT / folder
        if path.is_dir():
            print(f"exists   {folder}")
        else:
            path.mkdir(parents=True, exist_ok=True)
            print(f"created  {folder}")


if __name__ == "__main__":
    main()
