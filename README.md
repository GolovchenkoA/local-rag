# Local RAG

Ask questions about technical PDF books and get answers with sources. Everything runs locally on your laptop, nothing is sent to the cloud.

> This README is updated at every implementation step. Sections marked **🚧 Not implemented yet** describe the planned behaviour.
> The full list of project requirements is in [`RAG-requirements.md`](RAG-requirements.md).

## Implementation status

| Part | Status |
|---|---|
| Qdrant in Docker (persistent data) | ✅ Done (not yet tested on the target machine) |
| Docling server in Docker (GPU / CPU profiles, UI) | ✅ Done (not yet tested on the target machine) |
| Claude Code structure (`CLAUDE.md`, `.claude/skills/`) | ✅ Done |
| Python project, virtual environment, config file | ✅ Done (`pyproject.toml`, `config.yaml`, `.venv`) |
| Phase 1 pipeline: `parse` → `chunk` → `embed` (files in `data/`) | ✅ Code done; run on the first book (see section 4.2) |
| Phase 2: `upload-embeddings.ipynb` (upload to Qdrant) | ✅ Notebook written, **not run yet** (nothing uploaded) |
| Jupyter notebook for questions and answers (`notebooks/question-answering.ipynb`) | ✅ Written and tested end to end with `qwen3.5:9b` (see 5.1); the score threshold still needs tuning |

---

## 1. Project overview

**Pipeline** (each stage writes an inspectable file per book):

```
Phase 1 (Docling container only):  PDF ──parse──▶ Docling JSON + Markdown ──chunk──▶ chunks.jsonl ──embed──▶ vectors (files)
Phase 2 (Qdrant container only):   vectors + chunks ──upload-embeddings.ipynb──▶ Qdrant
                                                                                            │
question ──embed──▶ search top 10 ──▶ rerank ──▶ top 2 ──▶ Ollama (qwen3.5:9b) ──▶ answer + sources
```

**Components**

| Component | Role | Where it runs |
|---|---|---|
| Docling (`docling-serve`) | Parses PDFs into a structured document (headings, code blocks, notes, pages). Images are skipped, OCR is off. | Docker |
| Qdrant | Vector database. Stores vectors and metadata (`book_id`, `book_title`, `chapter`, `subchapter`, `pages`). | Docker |
| Embedding model and reranker | Turn text into vectors, reorder search results. | Python, CUDA with CPU fallback |
| Ollama | Runs the answering model `qwen3.5:9b`. | Native install |
| Jupyter notebook | Question field → search → rerank → answer with sources. | Python |

**Not committed to git:** models, embeddings, Qdrant data, PDFs, parsed files (see `.gitignore`). The repository holds only code and configuration.

**Data layout** (inside the gitignored `data/` folder):

```
data/
  pdfs/                                  source PDFs
  parsed/<book_id>/book.docling.json     full parse result (chunking works from this)
  parsed/<book_id>/book.md               readable export, for inspection only
  parsed/test/                           scratch folder for manual Docling API tests (raw responses)
  chunks/<book_id>/chunks.jsonl          one chunk per line: text + metadata (headings, position, ...)
  embeddings/<book_id>/                  vectors prepared for upload (format still to be decided)
  discussions/                           local design notes (for example embeddings.md)
```

Git does not track empty folders, so this structure does not exist after a clone. Create it with the script in section 4.0 (`py scripts/init_data.py`).

---

## 2. First configuration

### 2.1 Requirements

| Requirement | Notes |
|---|---|
| Python **3.14.7** | Used for the virtual environment |
| Docker with Docker Compose | Docker Desktop on Windows |
| NVIDIA driver 550.54.14 or newer | Only for the GPU profile |
| NVIDIA Container Toolkit (or Docker Desktop with WSL2 GPU support) | Only for the GPU profile |
| Ollama | For the answering model |
| Hardware used for development | i9-14900HX, 32 GB RAM, RTX 5060 Laptop (8 GB VRAM). 16 GB RAM is the practical minimum. |

Check what you have (PowerShell):

```powershell
py -3.14 --version          # expect Python 3.14.7
docker --version
docker compose version
nvidia-smi                  # GPU and driver version
ollama --version
```

### 2.2 Virtual environment

Dependencies are declared in [`pyproject.toml`](pyproject.toml). PyTorch is **not** listed there: it must be the CUDA 12.8 build (needed for the RTX 50-series), so install it first from the PyTorch index. Run from the project folder:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

```powershell
pip install -e ".[dev]"
```

`[dev]` adds JupyterLab, `nbstripout` and `pytest`. Check the GPU:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Install the notebook output stripper once per clone (notebooks are committed without outputs; `.gitattributes` is already in the repository):

```powershell
nbstripout --install --attributes .gitattributes
```

All components try CUDA first and fall back to the CPU if it is not available (`src/local_rag/device.py`, setting `device` in `config.yaml`). The chosen device is logged at startup. Run the tests with `pytest`.

### 2.3 Ollama models

```powershell
ollama pull qwen3.5:9b
```

The embedding model `BAAI/bge-m3` (about 2.2 GB) is downloaded by Python code from Hugging Face on the first `embed` run. The reranker `BAAI/bge-reranker-v2-m3` (about 2.3 GB) is downloaded the same way on the first run of the question answering notebook.

### 2.4 Configuration file

All settings live in [`config.yaml`](config.yaml): device, Docling URL and conversion switches (OCR off, images off, heading hierarchy on), chunking (tokenizer `BAAI/bge-m3`, max 600 tokens, no overlap), embedding model and batch size, Qdrant URL and collection names (`rag_books` main, `rag_test` for experiments), the **book list**, and the settings for the future question answering notebook (top-k 10, rerank top-n 2, score threshold 0.7, Ollama model, 1000 answer tokens, system prompt).

Each book is one entry:

```yaml
books:
  - file: book.pdf                 # file name in data/pdfs/
    book_id: ml-design-patterns    # stable slug, stored on every chunk
    book_title: Machine Learning Design Patterns
    skip_top_headings:             # chunks starting with these headings are not loaded
      - Machine Learning Design Patterns
      - Index
```

---

## 3. Run the Docker services

Qdrant and Docling are defined in [`docker-compose.yml`](docker-compose.yml). Run the commands from the project folder.

**Docling and Qdrant are not connected to each other.** Docling only converts PDFs into files. The embeddings are computed by the project's Python code and uploaded to Qdrant by a separate notebook. So the work is done in two separate phases, and each phase starts only the container it needs:

| Phase | What you do | Containers needed |
|---|---|---|
| 1. Prepare | Parse PDFs, chunk, compute embeddings (files in `data/`) | Docling (Qdrant not needed) |
| 2. Upload | Upload the prepared embeddings with `upload-embeddings.ipynb`, then ask questions | Qdrant (Docling not needed) |

### Option A: only Docling (phase 1: parse PDFs and prepare embeddings)

Choose one Docling profile:

```powershell
docker compose up -d docling-gpu       # NVIDIA GPU
```

```powershell
docker compose up -d docling-cpu       # CPU only
```

Only one of the two can run, because both use port 5001. Docling is needed by the `parse` and `chunk` stages, because they call the server (the `chunk` stage does not need it when the raw chunker response is already saved in `data/parsed/<book_id>/chunks.raw.json`). The `embed` stage runs in Python and needs no container. Stop Docling when these stages are done (see section 6).

> The first start downloads a large image (the CUDA Docling image is about 11 GB).

### Option B: only Qdrant (phase 2: upload embeddings and ask questions)

```powershell
docker compose up -d qdrant
```

Then run [`notebooks/upload-embeddings.ipynb`](notebooks/upload-embeddings.ipynb) to write the prepared chunks and vectors into Qdrant (section 4.3). No Docling image is needed for this.

### Option C: both at the same time

Only if you want to inspect conversions in the Docling UI while Qdrant is running:

```powershell
docker compose --profile gpu up -d     # or: --profile cpu
```

The `--profile` form starts Qdrant together with the chosen Docling service.

### Download the Docling image first (to watch the progress)

`up -d` runs in the background and shows little while it downloads. To see the layer-by-layer download progress, pull the image manually before the first start, in the foreground:

```powershell
docker compose --profile gpu pull docling-gpu      # or: --profile cpu pull docling-cpu
```

Each layer shows its own progress bar. It can take many minutes for the CUDA image. If it is interrupted, run the same command again: finished layers are kept and the download resumes. Then start the services with the `up -d` command above, which now starts immediately.

Check the downloaded images and their size:

```powershell
docker images
```

### Stop one service and keep the other

```powershell
docker compose stop docling-gpu        # or: docling-cpu
docker compose stop qdrant             # data is kept
```

### Check that everything is running

```powershell
docker compose ps
```

| Service | URL | Started by |
|---|---|---|
| Qdrant dashboard | http://localhost:6333/dashboard | Option B or C |
| Qdrant REST API | http://localhost:6333 | Option B or C |
| Docling UI | http://localhost:5001/ui | Option A or C |
| Docling API docs (Swagger) | http://localhost:5001/docs | Option A or C |

All ports are bound to localhost only.

### Data persistence

Qdrant data lives in the named Docker volumes `local-rag-qdrant-storage` and `local-rag-qdrant-snapshots`. It survives `docker stop` / `docker start` and `docker compose down` / `up`. Only `docker compose down -v` deletes it.

### If something does not start

- **Docling GPU image fails to pull:** the pinned tag may not exist for the `cu128` image. Check the image table in the docling-serve repository, then change the `image:` line in `docker-compose.yml`.
- **GPU not visible in the container:** check that the NVIDIA Container Toolkit (or WSL2 GPU support in Docker Desktop) is installed, or use the `cpu` profile.
- **Port already in use:** another process uses 6333, 6334 or 5001.

---

## 4. Upload a PDF and validate the result

### 4.0 Create the `data/` folders and put your PDFs in `data/pdfs/`

The whole `data/` folder is gitignored (together with all `*.pdf` files), so it exists only on your machine and is missing after a fresh clone. Create the folder structure (see the data layout in section 1) with the script [`scripts/init_data.py`](scripts/init_data.py). It creates only the folders that are absent, so it is safe to run again at any time, and it needs no virtual environment (standard library only):

```powershell
py scripts/init_data.py
```

It prints `created` or `exists` for each folder.

The pipeline will create the per-book folders (`data/parsed/<book_id>/`, `data/chunks/<book_id>/`, `data/embeddings/<book_id>/`) by itself when it runs. `data/parsed/test/` is only for manual tests, for example saving a raw Docling API response.

Then copy the source books into `data/pdfs/`:

```
data/pdfs/<book>.pdf
```

The `parse` stage (not implemented yet) will read the PDFs from this folder. The `book_id` is set in the config file, or derived from the file name.

### 4.1 Check the conversion in the Docling UI (available now)

Use this to see how a PDF is converted before running the whole pipeline.

1. Open http://localhost:5001/ui.
2. Upload a PDF (a single chapter is best for a quick test).
3. Read the Markdown result and check:
   - Headings are detected on the right levels (chapter, sub-chapter).
   - Code blocks are intact and keep their indentation.
   - Notes are not mixed into the neighbouring text.
   - Page headers and footers did not leak into the text.
   - Reading order is correct.

The full JSON (with page numbers) is also available through the API docs at http://localhost:5001/docs.

### 4.2 Run the pipeline

Each stage works on one book or on all books and is skipped for books that already have its output, unless forced with `--force`. Run from the project folder with the virtual environment active:

```powershell
python -m local_rag <stage> --book <book_id|all> [--force]
```

**Phase 1: prepare (only Docling is running, Option A in section 3).** Files only, nothing is written to Qdrant.

| Stage | What it does | Output to inspect |
|---|---|---|
| `parse` | PDF → Docling JSON and Markdown (calls the Docling container) | `data/parsed/<book_id>/book.md` |
| `chunk` | The Docling hybrid chunker (max 600 tokens, headings in the text) is called, then the result is post-processed: project fields added, front matter and index dropped (`skip_top_headings`), callout labels `TIP`/`NOTE`/`WARNING` kept. Raw response saved as `chunks.raw.json`. | `data/chunks/<book_id>/chunks.jsonl` |
| `embed` | Dense vectors for every chunk with `bge-m3` (CUDA with CPU fallback). Needs no container. | `data/embeddings/<book_id>/dense.npy` and `meta.json` |

Example, for the book configured in `config.yaml` (the PDF is `data/pdfs/book.pdf`):

```powershell
docker compose up -d docling-gpu
```

```powershell
python -m local_rag parse --book ml-design-patterns
```

```powershell
python -m local_rag chunk --book ml-design-patterns
```

```powershell
docker compose stop docling-gpu
```

```powershell
python -m local_rag embed --book ml-design-patterns
```

What the files contain:
- **`chunks.jsonl`**: one chunk per line with `id` (UUIDv5 from `book_id` + heading path + position), `book_id`, `book_title`, `chapter`, `headings` (full path), `pages` (PDF page numbers), `position`, `token_count`, `type`, `chunker_version` and `text` (headings + body).
- **`dense.npy`**: float32 matrix, one 1024-dimension vector per chunk, in the same order as `chunks.jsonl`.
- **`meta.json`**: the model name, dimension, count, chunk IDs in order and a hash of `chunks.jsonl`. It lets `embed` skip finished work and lets the upload notebook refuse vectors that do not match the chunks.

Recommended order: process **one book first**, inspect the `.md` and `chunks.jsonl`, fix settings, and only then run all books.

**Phase 2: upload (only Qdrant is running, Option B in section 3).**

| Step | What it does | Output to inspect |
|---|---|---|
| `notebooks/upload-embeddings.ipynb` | Reads the prepared chunks and vectors from `data/`, creates the collection and uploads | Qdrant dashboard |

### 4.3 Upload to Qdrant

Start only Qdrant (`docker compose up -d qdrant`), then open the notebook in JupyterLab. Use `python -m jupyter` (with the virtual environment active) instead of the `jupyter.exe` launcher: on some Windows machines the Application Control policy blocks that `.exe` (error `WinError 4551`), while `python.exe` is allowed:

```powershell
python -m jupyter lab notebooks/upload-embeddings.ipynb
```

**Jupyter fails with `PermissionError ... AppData\Roaming\jupyter\runtime\jpserver-...-open.html`, or the notebook shows "queued" in PyCharm:** Jupyter cannot write to its runtime folder `C:\Users\<you>\AppData\Roaming\jupyter\runtime` (on one machine that folder had broken permissions: even `ls` was denied). Point Jupyter at a folder you own. For one terminal session:

```powershell
$env:JUPYTER_RUNTIME_DIR = "$env:LOCALAPPDATA\jupyter\runtime"
python -m jupyter lab --no-browser
```

To make it permanent for your Windows user (restart PyCharm afterwards):

```powershell
setx JUPYTER_RUNTIME_DIR "%LOCALAPPDATA%\jupyter\runtime"
```

Alternatively repair the permissions of the original folder from an administrator PowerShell (`takeown /f "$env:APPDATA\jupyter" /r /d y`, then `icacls "$env:APPDATA\jupyter" /grant "$env:USERNAME:(OI)(CI)F" /t`). The kernel itself works with this `.venv` (checked headless: it starts in about 2 seconds).

The notebook targets the **test collection** (`rag_test`) by default and has a safety flag: the upload cell runs with `DRY_RUN = True`, which only checks that the chunks and vectors match and reports what would be uploaded. Set `DRY_RUN = False` to write. The collection gets the named vector `dense` (1024 dimensions, cosine) and payload indexes on `book_id` and `position`. Re-running replaces the book's points, so changed chunking never leaves old points behind.

### 4.4 Check the data in Qdrant

1. Open http://localhost:6333/dashboard.
2. Open **Collections** and select the collection.
3. Check the number of points, and open a few points to see their payload: `book_id`, `book_title`, `chapter`, `subchapter`, `pages`, `text`.

---

## 5. Run the project

### 5.1 Jupyter notebook (questions and answers)

The notebook is [`notebooks/question-answering.ipynb`](notebooks/question-answering.ipynb). The code is in [`src/local_rag/qa.py`](src/local_rag/qa.py) and every value comes from the `qa` section of [`config.yaml`](config.yaml).

**Before you start**
1. Qdrant is running and the collection is filled (`docker compose up -d qdrant`, then `upload-embeddings.ipynb`, section 4.3). Docling is not needed.
2. Ollama is running and the model is pulled: `ollama pull qwen3.5:9b` (6.6 GB).
3. Open the notebook (see section 4.3 for the `python -m jupyter lab` command and the Windows notes).

**What it does when you run a question** (set `question = "..."` in section 2 of the notebook and run the cell)

1. The question is embedded with `bge-m3` (dense vector, same model as the chunks).
2. Qdrant returns the **top 10** most similar chunks.
3. The reranker (`BAAI/bge-reranker-v2-m3`) scores the 10 chunks; the score is normalized to 0-1.
4. The **top 2** chunks are kept as context. Optionally `neighbor_window` adds the chunks before and after each one when they are in the same section (default 0 = off).
5. If the best score is below `score_threshold` the answer is *"The answer is not in the books."* and the model is **not** called.
6. Otherwise the question and the chunks go to the Ollama model (`num_predict: 1000` tokens). The system prompt tells the model to use only the chunks, never guess, and say so when the answer is missing.

The notebook shows an `answer`, the `sources` (built from the chunk metadata: `book_title`, `book_id`, `chapter`, `headings`, `pages`, not written by the model) and a table of all 10 searched chunks with their rerank and vector scores. The last cell prints the full text of the chunks behind the last answer.

**Tested** (first book, `rag_test` collection, query embedder and reranker on the CPU, model `qwen3.5:9b`, test run with the threshold lowered to 0.3):
- Search plus rerank take about 6 seconds. The model call took about 30 seconds the first time (the model is loaded into memory) and about 6 seconds afterwards.
- "What are CNNs?" and "How can I handle a categorical feature with very many distinct values?" both got answers grounded in the two chunks (prompts of 871 and 1021 tokens, answers of 101 and 147 tokens, finished normally, so the 1000-token limit was not reached). `think: false` is accepted.
- An off-topic question ("Who won the football world cup in 2018?") scored 0.002 and was answered "The answer is not in the books." without calling the model.
- The model starts its answer with "answer:" and sometimes adds its own "Sources: Chunk [1], [2]" line (the system prompt says the response should contain "answer" and a source list). The `sources` list built from the metadata is the reliable one.

**Known issue: the 0.7 threshold is too strict for this reranker.** Questions that the book does answer scored 0.68 ("What are CNNs?") and 0.52 ("How can I handle a categorical feature with very many distinct values?"), so both would be answered "not in the books", while the off-topic question scored 0.002. The gap suggests a threshold of roughly 0.3 to 0.4, but the value should be chosen with a set of test questions (open question #15). Change `qa.score_threshold` in `config.yaml`.

### 5.2 Tips for the 8 GB GPU

- Unload the chat model while ingesting books, so Docling and the embedder get the whole GPU: `ollama stop qwen3.5:9b`.
- When asking questions, the query embedder and reranker run on the CPU by default (`qa.embedder_device`, `qa.reranker_device`), which leaves the VRAM to the answering model.

---

## 6. Stop all services

Stop containers and **keep all data**:

```powershell
docker compose --profile gpu down      # or: --profile cpu
```

Pause without removing containers (data is kept as well):

```powershell
docker compose --profile gpu stop
docker compose --profile gpu start
```

Unload the Ollama model:

```powershell
ollama stop qwen3.5:9b
```

Stop Jupyter with `Ctrl+C` in its terminal, then `deactivate` the virtual environment.

> ⚠️ **Do not use `docker compose down -v`** unless you want to delete the whole Qdrant database. You would have to run `load` again for every book.
