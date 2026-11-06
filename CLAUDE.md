# Local RAG – instructions for Claude

Full requirements: @RAG-requirements.md. User-facing docs: @README.md.

## Working rules
- Build the project **step by step**. After every step update `README.md` (status table + the relevant section).
- This folder is the git root. Commit only source code and configuration. Never commit `data/`, models, embeddings, PDFs, `.venv/` (enforced by `.gitignore`).
- Python 3.14.7 in an isolated `.venv`. Never install packages globally.
- Every GPU-capable component uses CUDA first with CPU fallback, through one shared device helper. Log the chosen device at startup.
- All tunable values (top-k 10, rerank top-n 2, model name, token limit 1000, system prompt, OCR/image switches, device) live in the config file, never hard-coded.
- Pin Docker image versions. Never use `latest`. Never run `docker compose down -v` (deletes the Qdrant data).
- Experiments and write tests use a separate test Qdrant collection, not the main one.
- Notebooks are committed with outputs stripped (`nbstripout`).

## Stack
Qdrant (Docker) · docling-serve (Docker, `gpu`/`cpu` profiles) · `bge-m3` dense+sparse embeddings · reranker · Ollama `qwen3.5:9b` · JupyterLab.

## Environment
Windows 11, PowerShell. Target hardware: i9-14900HX, 32 GB RAM, RTX 5060 Laptop 8 GB (PyTorch CUDA 12.8+).

## Project skills
Project-specific skills live in `.claude/skills/<name>/SKILL.md` and are committed. See `.claude/skills/README.md`.
