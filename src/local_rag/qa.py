"""Question answering over Qdrant.

Flow: embed the question (bge-m3, dense) -> vector search (top 10) -> rerank ->
keep the best chunks (top 2) -> if the best score is below the threshold, answer
"not found" -> otherwise ask the Ollama model with only those chunks as context.

Everything tunable comes from config.yaml (qa section). Nothing runs on import.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable

import httpx
import numpy as np
from qdrant_client import QdrantClient, models

from . import embedding
from .device import resolve_device

log = logging.getLogger(__name__)


@dataclass
class Hit:
    """One chunk returned by the vector search."""
    id: str
    vector_score: float          # cosine similarity from Qdrant
    payload: dict
    rerank_score: float | None = None   # normalized 0-1, set by rerank()
    primary: bool = True         # False for neighbour chunks added by expand_neighbors()

    @property
    def position(self) -> int:
        return self.payload["position"]


# --- models ---------------------------------------------------------------------

def load_query_embedder(cfg: SimpleNamespace):
    """The same bge-m3 model as for the chunks (device from qa.embedder_device)."""
    return embedding.load_model(cfg, cfg.qa.embedder_device)


def load_reranker(cfg: SimpleNamespace):
    """Cross-encoder reranker (device from qa.reranker_device)."""
    from sentence_transformers import CrossEncoder

    device = resolve_device(cfg.qa.reranker_device)
    return CrossEncoder(cfg.qa.reranker_model, device=device,
                        max_length=cfg.qa.rerank_max_length)


# --- retrieval ------------------------------------------------------------------

def embed_question(model, cfg: SimpleNamespace, question: str) -> list[float]:
    return embedding.embed_texts(model, cfg, [question])[0].tolist()


def search(client: QdrantClient, cfg: SimpleNamespace, collection: str,
           query_vector: list[float], top_k: int | None = None) -> list[Hit]:
    result = client.query_points(
        collection_name=collection,
        query=query_vector,
        using=cfg.qdrant.vector_name,
        limit=top_k or cfg.qa.retrieve_top_k,
        with_payload=True,
    )
    return [Hit(id=str(p.id), vector_score=p.score, payload=p.payload) for p in result.points]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def rerank(reranker, question: str, hits: list[Hit]) -> list[Hit]:
    """Score every (question, chunk) pair, store a 0-1 score, return hits best first."""
    if not hits:
        return []
    logits = np.asarray(reranker.predict([(question, h.payload["text"]) for h in hits],
                                         apply_softmax=False, convert_to_numpy=True),
                        dtype=np.float64)
    # bge-reranker outputs one raw score (logit) per pair; sigmoid maps it to 0-1.
    # If the library already returned probabilities, keep them.
    scores = logits if (logits.min() >= 0.0 and logits.max() <= 1.0) else _sigmoid(logits)
    for hit, score in zip(hits, scores):
        hit.rerank_score = float(score)
    return sorted(hits, key=lambda h: h.rerank_score, reverse=True)


def select_context(ranked: list[Hit], top_n: int, threshold: float) -> tuple[list[Hit], bool]:
    """Keep the top_n chunks. Second value is True when the best score is below the threshold."""
    if not ranked:
        return [], True
    best = ranked[0].rerank_score
    below = best is None or best < threshold
    return ranked[:top_n], below


# --- neighbours (optional, qa.neighbor_window > 0) --------------------------------

def merge_context(selected: list[Hit], neighbours: list[Hit]) -> list[Hit]:
    """Selected chunks plus neighbours, without duplicates, in reading order (position)."""
    by_id: dict[str, Hit] = {h.id: h for h in neighbours}
    by_id.update({h.id: h for h in selected})        # selected chunks win over neighbours
    return sorted(by_id.values(), key=lambda h: (h.payload["book_id"], h.position))


def expand_neighbors(client: QdrantClient, cfg: SimpleNamespace, collection: str,
                     selected: list[Hit], window: int) -> list[Hit]:
    """Add the chunks at position +/- window that belong to the same section (same headings)."""
    if window <= 0 or not selected:
        return selected
    neighbours: list[Hit] = []
    for hit in selected:
        lo, hi = hit.position - window, hit.position + window
        points, _ = client.scroll(
            collection_name=collection, limit=2 * window + 1, with_payload=True,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="book_id",
                                      match=models.MatchValue(value=hit.payload["book_id"])),
                models.FieldCondition(key="position", range=models.Range(gte=lo, lte=hi)),
            ]))
        for p in points:
            if p.payload["headings"] == hit.payload["headings"]:      # same section only
                neighbours.append(Hit(id=str(p.id), vector_score=0.0, payload=p.payload,
                                      primary=False))
    return merge_context(selected, neighbours)


# --- answer ---------------------------------------------------------------------

def build_messages(cfg: SimpleNamespace, question: str, context: list[Hit]) -> list[dict]:
    blocks = []
    for i, hit in enumerate(context, start=1):
        p = hit.payload
        pages = ", ".join(str(x) for x in p["pages"])
        blocks.append(f"[{i}] {p['book_title']} | {' > '.join(p['headings'])} | pages {pages}\n"
                      f"{p['text']}")
    user = ("Retrieved chunks:\n\n" + "\n\n".join(blocks)
            + f"\n\nQuestion: {question}")
    return [{"role": "system", "content": cfg.qa.system_prompt.strip()},
            {"role": "user", "content": user}]


def sources_from(context: list[Hit]) -> list[dict]:
    """Source list built from the chunk metadata (not written by the model)."""
    out, seen = [], set()
    for hit in context:
        p = hit.payload
        key = (p["book_id"], tuple(p["headings"]), tuple(p["pages"]))
        if key in seen:
            continue
        seen.add(key)
        out.append({"book_title": p["book_title"], "book_id": p["book_id"],
                    "chapter": p["chapter"], "headings": p["headings"], "pages": p["pages"]})
    return out


def ask_ollama(cfg: SimpleNamespace, messages: list[dict],
               post: Callable = httpx.post) -> dict:
    o = cfg.qa.ollama
    resp = post(
        o.url.rstrip("/") + "/api/chat",
        json={"model": o.model, "messages": messages, "stream": False, "think": o.think,
              "options": {"num_predict": o.num_predict, "num_ctx": o.num_ctx}},
        timeout=o.timeout_seconds,
    )
    resp.raise_for_status()
    data = resp.json()
    return {"text": data["message"]["content"].strip(),
            "prompt_tokens": data.get("prompt_eval_count"),
            "answer_tokens": data.get("eval_count"),
            "done_reason": data.get("done_reason")}


# Two empty lines in rendered Markdown (plain blank lines would collapse into one).
SPACER = "\n\n&nbsp;\n\n&nbsp;\n\n"
_LABEL = re.compile(r"(\(\s*)?\b(answers?|sources?)\s*:[ \t]*", re.IGNORECASE)


def format_answer(text: str) -> str:
    """Display form of the model's reply: after "answer:" and "source(s):" the text starts on
    a new line, and "Sources:" is separated from the answer by two empty lines."""
    text = text.strip()
    had_paren = False

    def repl(m: re.Match) -> str:
        nonlocal had_paren
        label = "Answer" if m.group(2).lower().startswith("answer") else "Sources"
        had_paren = had_paren or bool(m.group(1))
        if m.start() == 0:
            head = ""
        else:
            head = SPACER if label == "Sources" else "\n\n"
        return f"{head}**{label}:**\n\n"

    out = _LABEL.sub(repl, text)
    if had_paren and out.endswith(")"):
        out = out[:-1].rstrip()          # the "(" before "Source:" was removed, drop its ")"
    out = re.sub(r"[ \t]+\n", "\n", out)
    return re.sub(r"\n{3,}", "\n\n", out)      # blank lines are made by SPACER, not by repeats


@dataclass
class Result:
    question: str
    answer: str
    sources: list[dict]
    found: bool                      # False = best chunk below the score threshold
    candidates: list[Hit] = field(default_factory=list)   # all searched chunks, best first
    context: list[Hit] = field(default_factory=list)      # chunks sent to the model
    timings: dict = field(default_factory=dict)
    llm: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        """The response shown to the user: an answer and a list of sources."""
        return {"answer": self.answer, "sources": self.sources}


def answer_question(question: str, cfg: SimpleNamespace, client: QdrantClient,
                    collection: str, embedder, reranker) -> Result:
    qa = cfg.qa
    t = {}
    t0 = time.perf_counter()
    vector = embed_question(embedder, cfg, question)
    t["embed"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    hits = search(client, cfg, collection, vector, qa.retrieve_top_k)
    t["search"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ranked = rerank(reranker, question, hits)
    t["rerank"] = time.perf_counter() - t0

    selected, below = select_context(ranked, qa.rerank_top_n, qa.score_threshold)
    if below:
        log.info("Best rerank score %.3f is below the threshold %.2f: not found",
                 ranked[0].rerank_score if ranked else float("nan"), qa.score_threshold)
        return Result(question, qa.not_found_message, [], False, ranked, [], t)

    context = expand_neighbors(client, cfg, collection, selected, qa.neighbor_window)
    t0 = time.perf_counter()
    llm = ask_ollama(cfg, build_messages(cfg, question, context))
    t["llm"] = time.perf_counter() - t0
    return Result(question, llm["text"], sources_from(selected), True, ranked, context, t, llm)
