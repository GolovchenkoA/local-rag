# Local RAG

Ask questions about technical PDF books and get answers with sources. Everything runs locally on your laptop, nothing is sent to the cloud.

> This README is updated at every implementation step. Sections marked **🚧 Not implemented yet** describe the planned behaviour.
> The full list of project requirements is in [`RAG-requirements.md`](RAG-requirements.md).

## Implementation status

| Part | Status |
|---|---|
| Qdrant in Docker (persistent data) | ✅ Done (not yet tested on the target machine) |
| Docling server in Docker (GPU / CPU profiles, UI) | ✅ Done (not yet tested on the target machine) |
| Python project, virtual environment, config file | 🚧 Not implemented yet |
| Pipeline: `parse` → `chunk` → `embed` → `load` | 🚧 Not implemented yet |
| Jupyter notebook for questions and answers | 🚧 Not implemented yet |

---

## 1. Project overview

**Pipeline** (each stage writes an inspectable file per book):

```
PDF ──parse──▶ Docling JSON + Markdown ──chunk──▶ chunks.jsonl ──embed──▶ vectors ──load──▶ Qdrant
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
  chunks/<book_id>/chunks.jsonl          one chunk per line: text + metadata
  embeddings/<book_id>/                  optional vector cache
```

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

### 2.2 Virtual environment 🚧 Not implemented yet

The dependency file does not exist yet. When it is added, the setup will be:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
# pip install ...   (command will be added with the dependency file)
```

PyTorch must be a CUDA 12.8+ build (needed for the RTX 50-series). All components try CUDA first and fall back to the CPU if it is not available.

### 2.3 Ollama models

```powershell
ollama pull qwen3.5:9b
```

The embedding model and the reranker are downloaded by Python code on first use (🚧 not implemented yet).

### 2.4 Configuration file 🚧 Not implemented yet

All settings (book list with `book_id` and `book_title`, top-k = 10, rerank top-n = 2, model name, token limit = 1000, system prompt, device, OCR and image switches) will live in one config file. This section will describe it.

---

## 3. Run the Docker services

Qdrant and Docling are defined in [`docker-compose.yml`](docker-compose.yml). Run the commands from the project folder.

### Option A: with the NVIDIA GPU

```powershell
docker compose --profile gpu up -d
```

### Option B: CPU only

```powershell
docker compose --profile cpu up -d
```

Start **only one** of the two profiles, because both Docling services use port 5001. Qdrant starts with either command.

> The first start downloads large images (the CUDA Docling image is about 11 GB).

### Check that everything is running

```powershell
docker compose ps
```

| Service | URL |
|---|---|
| Qdrant dashboard | http://localhost:6333/dashboard |
| Qdrant REST API | http://localhost:6333 |
| Docling UI | http://localhost:5001/ui |
| Docling API docs (Swagger) | http://localhost:5001/docs |

All ports are bound to localhost only.

### Data persistence

Qdrant data lives in the named Docker volumes `local-rag-qdrant-storage` and `local-rag-qdrant-snapshots`. It survives `docker stop` / `docker start` and `docker compose down` / `up`. Only `docker compose down -v` deletes it.

### If something does not start

- **Docling GPU image fails to pull:** the pinned tag may not exist for the `cu128` image. Check the image table in the docling-serve repository, then change the `image:` line in `docker-compose.yml`.
- **GPU not visible in the container:** check that the NVIDIA Container Toolkit (or WSL2 GPU support in Docker Desktop) is installed, or use the `cpu` profile.
- **Port already in use:** another process uses 6333, 6334 or 5001.

---

## 4. Upload a PDF and validate the result

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

### 4.2 Run the pipeline 🚧 Not implemented yet

Planned stages, each working on one book or all books, and each skipped for books that already have its output unless forced:

| Stage | What it does | Output to inspect |
|---|---|---|
| `parse` | PDF → Docling JSON and Markdown | `data/parsed/<book_id>/book.md` |
| `chunk` | Structure-aware chunks with metadata | `data/chunks/<book_id>/chunks.jsonl` |
| `embed` | Vectors for each chunk | `data/embeddings/<book_id>/` |
| `load` | Write chunks and vectors to Qdrant | Qdrant dashboard |

Recommended order: process **one chapter or one book first**, inspect the `.md` and `chunks.jsonl`, fix settings, and only then run all books.

### 4.3 Check the data in Qdrant (after `load` exists)

1. Open http://localhost:6333/dashboard.
2. Open **Collections** and select the collection.
3. Check the number of points, and open a few points to see their payload: `book_id`, `book_title`, `chapter`, `subchapter`, `pages`, `text`.

---

## 5. Run the project

### 5.1 Jupyter notebook (questions and answers) 🚧 Not implemented yet

Planned flow:

1. Start the Docker services (section 3) and Ollama.
2. Activate the virtual environment and start JupyterLab.
3. Open the notebook and type a question in the query field.
4. The notebook retrieves the top 10 chunks, reranks them, keeps the top 2, and asks `qwen3.5:9b` (answer limited to 1000 tokens).
5. The result contains an `answer` and a list of `sources` (from the chunk metadata).

### 5.2 Tips for the 8 GB GPU

- Unload the chat model while ingesting books, so Docling and the embedder get the whole GPU: `ollama stop qwen3.5:9b`.
- When asking questions, the query embedder and reranker can run on the CPU, which leaves the VRAM to the answering model.

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
