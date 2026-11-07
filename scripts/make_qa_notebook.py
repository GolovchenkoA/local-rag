"""(Re)generate notebooks/question-answering.ipynb. The notebook is committed without outputs.

    .venv\\Scripts\\python.exe scripts/make_qa_notebook.py
"""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "question-answering.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


CELLS = [
    md("""
# Question answering over the books

Set the `question` text in section 2 and run the cell. The notebook then runs this flow (all values come from the `qa` section of `config.yaml`):

1. embed the question with `bge-m3` (dense vector, same model as the chunks);
2. search Qdrant and take the **top 10** most similar chunks;
3. **rerank** the 10 chunks with the reranker model;
4. keep the **top 2** reranked chunks as context;
5. if the best reranked score is below the threshold (**0.7**), answer *"not in the books"* without calling the model;
6. otherwise send the question and the chunks to the Ollama model (answer limited to 1000 tokens).

The result is an `answer` and a list of `sources` built from the metadata of the chunks used as context.

**Needed:** Qdrant running with the collection uploaded (`docker compose up -d qdrant`, then
`upload-embeddings.ipynb`), and Ollama running with the model pulled (`ollama pull qwen3.5:9b`).
"""),
    code("""
import logging

from local_rag.config import load_config
from local_rag import qa, qdrant_store

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
cfg = load_config()

# Which collection to ask. Keep the test collection until the main one is filled.
COLLECTION = cfg.qdrant.test_collection      # cfg.qdrant.collection = main collection

print("Collection        :", COLLECTION)
print("Top-k / top-n     :", cfg.qa.retrieve_top_k, "/", cfg.qa.rerank_top_n)
print("Score threshold   :", cfg.qa.score_threshold)
print("Neighbour window  :", cfg.qa.neighbor_window, "(0 = off)")
print("Model             :", cfg.qa.ollama.model, "| answer limit", cfg.qa.ollama.num_predict, "tokens")
"""),
    md("""
## 1. Connect and load the models (once)

The query embedder and the reranker run on the CPU by default (`qa.embedder_device`, `qa.reranker_device`),
which leaves the GPU memory to the Ollama model. The first run downloads the reranker from Hugging Face.
"""),
    code("""
import httpx

client = qdrant_store.connect(cfg)
count = client.count(COLLECTION, exact=True).count
print(f"Qdrant: collection {COLLECTION} has {count} points")

embedder = qa.load_query_embedder(cfg)
reranker = qa.load_reranker(cfg)

# Is the Ollama model available?
try:
    tags = httpx.get(cfg.qa.ollama.url.rstrip("/") + "/api/tags", timeout=10).json()
    names = [m["name"] for m in tags["models"]]
    if cfg.qa.ollama.model in names:
        print("Ollama: model", cfg.qa.ollama.model, "is available")
    else:
        print(f"Ollama is running but the model {cfg.qa.ollama.model!r} is not pulled. "
              f"Run: ollama pull {cfg.qa.ollama.model}")
except httpx.HTTPError as e:
    print("Ollama is not reachable at", cfg.qa.ollama.url, "-", e)
"""),
    md("""
### Helper: run the flow and show the result

`ask()` runs the whole flow and shows the answer, the sources and a table of the 10 searched chunks
(rerank score, vector score, section, pages; `used` marks the chunks sent to the model).
"""),
    code("""
from IPython.display import Markdown, display

last_result = None


def ask(question: str):
    global last_result
    result = qa.answer_question(question, cfg, client, COLLECTION, embedder, reranker)
    last_result = result

    lines = [f"**Question:** {question}", "", "### Answer", "", result.answer, ""]
    if result.found:
        lines += ["### Sources", ""]
        for s in result.sources:
            pages = ", ".join(str(p) for p in s["pages"])
            lines.append(f"- *{s['book_title']}* (`{s['book_id']}`) - {s['chapter']} > "
                         f"{' > '.join(s['headings'][1:]) or '-'} - pages {pages}")
    else:
        best = result.candidates[0].rerank_score if result.candidates else 0.0
        lines += [f"*Best reranked score {best:.3f} is below the threshold "
                  f"{cfg.qa.score_threshold}; the model was not called.*"]

    used = {h.id for h in result.context}
    lines += ["", "### Searched chunks (best first)", "",
              "| # | rerank | vector | section | pages | used |", "|---|---|---|---|---|---|"]
    for i, h in enumerate(result.candidates, start=1):
        section = " > ".join(h.payload["headings"][-2:]).replace("|", "/")
        pages = ", ".join(str(p) for p in h.payload["pages"])
        lines.append(f"| {i} | {h.rerank_score:.3f} | {h.vector_score:.3f} | {section} | "
                     f"{pages} | {'yes' if h.id in used else ''} |")

    t = result.timings
    lines += ["", "*" + ", ".join(f"{k} {v:.1f}s" for k, v in t.items()) + "*"]
    if result.llm:
        lines[-1] = lines[-1][:-1] + (f", prompt {result.llm['prompt_tokens']} tokens, answer "
                                      f"{result.llm['answer_tokens']} tokens "
                                      f"({result.llm['done_reason']})*")
    display(Markdown("\\n".join(lines)))
    return result
"""),
    md("""
## 2. Ask

Change the text of `question` and run the cell again.
"""),
    code("""
question = "What are CNNs?"

result = ask(question)
"""),
    md("""
## 3. Read the chunks behind the last answer

Shows the text of the chunks that were sent to the model (`used`) and, optionally, all searched chunks.
"""),
    code("""
SHOW_ALL_SEARCHED = False      # True: show all 10 searched chunks, not only the ones sent to the model

if last_result is None:
    print("Ask a question first.")
else:
    chunks = last_result.candidates if SHOW_ALL_SEARCHED else last_result.context
    for h in chunks:
        p = h.payload
        score = f"{h.rerank_score:.3f}" if h.rerank_score is not None else "neighbour"
        print("=" * 100)
        print(f"score {score} | pages {p['pages']} | position {p['position']} | "
              f"{' > '.join(p['headings'])}")
        print("-" * 100)
        print(p["text"])
"""),
]

for _i, _cell in enumerate(CELLS):
    _cell["id"] = f"cell-{_i:02d}"

NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (local-rag)", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(NOTEBOOK, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", OUT)
