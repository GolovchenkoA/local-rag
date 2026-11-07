# Local RAG – Project Requirements

## Role
Act as an experienced Python developer who uses isolated environments. You are also an experienced RAG developer.

## Repository rules
- This folder is the root of the whole git project.
- The repository must contain only source code and configuration files.
- Never commit models, embeddings, vector-database data, parsed/cached documents, source PDFs, virtual environments or any other generated/large data (enforced via `.gitignore`).

## Python
- Use Python 3.14.7.
- Use an isolated virtual environment; dependencies are declared in the project, never installed globally.

## Vector database: Qdrant (Docker)
- Qdrant runs in a Docker container defined in `docker-compose.yml`.
- All Qdrant data (collections and snapshots) must survive `docker stop` / `docker start`
  (and `docker compose down` / `up`). Data is kept in named Docker volumes, outside the git repository.
- Ports are bound to localhost only: 6333 (HTTP/REST + dashboard), 6334 (gRPC).
- The Qdrant image version is pinned (no `latest`).

## Hardware acceleration: CUDA with CPU fallback
- Every component that can use a GPU (PDF parsing with Docling, embedding model, reranker, any PyTorch code) must try CUDA first and automatically fall back to CPU if CUDA is not available.
- Device selection is done in one place (a shared helper / config setting, e.g. `device: auto | cuda | cpu`) and the chosen device is logged at startup.
- Ollama manages its own GPU usage (uses CUDA when available, otherwise CPU).
- Target machine: Intel i9-14900HX, 32 GB RAM, NVIDIA GeForce RTX 5060 Laptop (8 GB VRAM) — PyTorch must be a CUDA 12.8+ build (Blackwell support).

## PDF parsing (Docling)
- Source PDFs are born-digital (they contain a text layer), so OCR is disabled (`do_ocr: false`) by default.
- OCR must stay configurable in the config file (engine and languages), so it can be switched on later for scanned PDFs or code embedded as images. Preferred engine in that case: EasyOCR (CUDA with CPU fallback), explicit languages, never `auto`.
- Images and diagrams in the PDFs are skipped during parsing: no image extraction, no OCR of bitmaps, no picture description. Only the text layer is used (this includes figure captions and the text around the figures).
- These switches stay configurable in the config file, all off by default: `extract_images`, `ocr_bitmaps`, `describe_pictures`.

## Multiple books and intermediate files
- Several PDFs (up to ~10 books, ~8,000 pages in total) are loaded into the same Qdrant collection.
- Every book has a stable `book_id` (slug set in the config file, or derived from the file name). It is stored as metadata (payload) on every chunk, together with `book_title` (human-readable title, set in the config file, used for citations), `chapter` (first heading), `headings` (the full heading path as a list, any depth; headings also appear in the chunk text before embedding), `pages` and `position` (order of the chunk in the book, used to find neighbouring chunks). There is no `parent_id`. `book_id` and `position` are indexed fields in Qdrant (filter by book, fetch neighbours by range); `book_title` is stored but not indexed. The vector is stored as the named vector `dense`.
- Chunks are at most 600 tokens, counted with the `bge-m3` tokenizer (config value), with no overlap. Chunks follow section boundaries; neighbouring chunks are added at answer time by the notebook.
- Chunk IDs are deterministic (UUIDv5 from book_id + full heading path + position, fixed project namespace), so loading the same book twice updates existing points instead of duplicating them. Removing or replacing one book means deleting all points with that `book_id`.
- Docling and Qdrant are not connected to each other. Docling only converts PDFs into files; embeddings are computed by the project's own Python code and uploaded to Qdrant by a notebook.
- The work is done in two separate phases, and each phase needs only one container:
  1. **Prepare** (only Docling running): stages `parse`, `chunk`, `embed`, each working on one book or on all books. A stage skips books that already have its output, unless forced. Only `parse` calls Docling; `chunk` and `embed` run in Python without any container. Nothing is written to Qdrant in this phase.
  2. **Upload** (only Qdrant running): the notebook `upload-embeddings.ipynb` (the `load` step) reads the prepared chunks and vectors from `data/` and writes them to Qdrant. It also reuses the project's code and config, is idempotent (deterministic IDs, see above) and is committed with outputs stripped.
- `docker-compose.yml` must allow starting each service on its own (`docker compose up -d qdrant`, `docker compose up -d docling-gpu` / `docling-cpu`); Qdrant must not depend on Docling or the other way round.
- Each stage writes an inspectable intermediate file per book, so results can be reviewed before loading into Qdrant. All of them live in the gitignored `data/` folder:
  - `data/pdfs/` — source PDFs
  - `data/parsed/<book_id>/book.docling.json` — complete parse result (chunking works from this file)
  - `data/parsed/<book_id>/book.md` — readable Markdown export, for inspection only
  - `data/chunks/<book_id>/chunks.jsonl` — one chunk per line: text and metadata (including `headings` and `position`)
  - `data/embeddings/<book_id>/` — **required**: `dense.npy` (one float32 vector per chunk, same order as `chunks.jsonl`) and `meta.json` (model, dimension, count, chunk IDs, hash of `chunks.jsonl`); the upload notebook reads it and refuses vectors that do not match the chunks
  - `data/parsed/test/` — scratch folder for manual Docling API tests
  - `data/discussions/` — local design notes
- The `data/` folder is gitignored and therefore missing after a clone. The script `scripts/init_data.py` (standard library only, safe to re-run) creates the top-level folders if they are absent; the pipeline creates the per-book folders itself (creating parent folders as needed) when a stage writes its output.

## Jupyter notebook: question answering over the RAG
- JupyterLab is a dev dependency of the project's virtual environment. The notebook imports the project's own code and config (Qdrant URL, device helper, embedding and reranker wrappers) instead of duplicating it. Notebooks are committed with outputs stripped (e.g. `nbstripout`).
- The notebook has a cell with a plain string variable (`question = "..."`) where the user types the question (no input widget); running the cell runs this flow:
  1. Embed the query (same model as the chunks: `bge-m3`, **dense vectors only** for now; sparse vectors and fusion may be added later).
  2. Search Qdrant (dense search, named vector `dense`) and retrieve the **top 10** most similar chunks.
  3. **Rerank** the 10 results with the reranker model.
  4. Keep the **top 2** reranked chunks as context.
  5. Send the question and the 2 chunks to the Ollama model **`qwen3.5:9b`** to generate the answer.
- The model's answer is limited to **1000 tokens** (`num_predict: 1000`).
- System prompt of the model: `You are an experienced ML Engineer that answers questions from RAG. Response should contain "answer" and source list from the RAG metadata response`, extended with grounding rules. The model:
  - uses **only** the information from the retrieved chunks (the vector database) and must never guess;
  - may summarize or combine information from different chunks when appropriate;
  - must never add anything of its own that is not in the chunks;
  - says that the answer is not in the books when the chunks do not contain it.
- Score threshold: the reranker score is normalized to 0-1. If the best reranked chunk scores below **0.7**, the answer is "not found in the books" and the model is not asked to guess. The value is a config setting and can be tuned later with a test question set.
- Language: the books and the questions are English only.
- The notebook is `notebooks/question-answering.ipynb`; its logic is in `src/local_rag/qa.py`. The response is an `answer` plus the `sources`, built from the chunk metadata by code (not written by the model). The notebook also shows the searched chunks with their scores and the text of the chunks sent to the model.
- Optional `neighbor_window` setting (default 0 = off): also send the chunks at `position` +/- N that belong to the same section (same `headings` path).
- The response shown in the notebook contains an `answer` and a list of `sources`. The sources are built from the metadata of the 2 chunks used as context (`book_title`, `book_id`, `chapter`, `headings`, `pages`).
- Top-k retrieved (10), top-n after reranking (2), the score threshold (0.7), the model name, the token limit and the system prompt are set in the config file, not hard-coded in the notebook.
- Embedding model and reranker use CUDA with CPU fallback (shared device helper). The reranker and query embedder may run on the CPU to leave VRAM for the Ollama model.
- Experiments and write operations use a separate test collection, not the main one.

## Docling server (Docker)
- Docling runs as the official `docling-serve` container, defined in `docker-compose.yml` next to Qdrant, with two profiles (only one is started at a time, both use port 5001):
  - `gpu`: NVIDIA GPU, CUDA 12.8 image (`docling-serve-cu128`). Requires an NVIDIA driver and the NVIDIA Container Toolkit (or Docker Desktop with WSL2 GPU support).
  - `cpu`: CPU-only image (`docling-serve-cpu`), used when no GPU is available.
- Image versions are pinned (CUDA images have no `latest` tag).
- The API is bound to localhost only. Swagger docs are at `http://localhost:5001/docs`. The demo UI is enabled (`DOCLING_SERVE_ENABLE_UI=1`) at `http://localhost:5001/ui`, for inspecting how PDFs are converted.
- The project's Python code calls the server over HTTP to parse PDFs; chunking is done in the project's own code from the returned Docling JSON.
- Docling is started only for the `parse` phase and can be stopped afterwards; it never talks to Qdrant.
