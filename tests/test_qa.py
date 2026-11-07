from types import SimpleNamespace

from local_rag import qa
from local_rag.qa import Hit


def hit(position, score=None, headings=("Chapter 1", "Intro"), pages=(10,), text="body"):
    payload = {"book_id": "b", "book_title": "Book", "chapter": headings[0],
               "headings": list(headings), "pages": list(pages), "position": position,
               "text": text}
    return Hit(id=f"id{position}", vector_score=0.5, payload=payload, rerank_score=score)


CFG = SimpleNamespace(qa=SimpleNamespace(
    system_prompt="  SYSTEM PROMPT  ",
    ollama=SimpleNamespace(url="http://x:1/", model="m", num_predict=50, num_ctx=1000,
                           think=False, timeout_seconds=5)))


def test_select_context_keeps_top_n_and_flags_low_score():
    ranked = [hit(1, 0.9), hit(2, 0.8), hit(3, 0.1)]
    selected, below = qa.select_context(ranked, top_n=2, threshold=0.7)
    assert [h.position for h in selected] == [1, 2] and below is False

    selected, below = qa.select_context([hit(1, 0.69), hit(2, 0.2)], 2, 0.7)
    assert below is True and len(selected) == 2

    assert qa.select_context([], 2, 0.7) == ([], True)


def test_threshold_boundary_is_inclusive():
    _, below = qa.select_context([hit(1, 0.7)], 2, 0.7)
    assert below is False


def test_merge_context_orders_by_position_without_duplicates():
    selected = [hit(5, 0.9), hit(9, 0.8)]
    neighbours = [hit(4), hit(5), hit(6), hit(9), hit(10)]   # 5 and 9 already selected
    merged = qa.merge_context(selected, neighbours)
    assert [h.position for h in merged] == [4, 5, 6, 9, 10]
    assert merged[1].rerank_score == 0.9            # the selected chunk is kept, not the copy


def test_build_messages_contains_system_prompt_headings_pages_and_question():
    msgs = qa.build_messages(CFG, "What is X?", [hit(1, text="Chapter 1\nIntro\nX is Y.")])
    assert msgs[0] == {"role": "system", "content": "SYSTEM PROMPT"}
    user = msgs[1]["content"]
    assert "Book | Chapter 1 > Intro | pages 10" in user
    assert "X is Y." in user and user.endswith("Question: What is X?")


def test_sources_from_uses_metadata_and_drops_duplicates():
    context = [hit(1, pages=(10,)), hit(2, pages=(10,)), hit(3, pages=(11, 12))]
    sources = qa.sources_from(context)
    assert len(sources) == 2
    assert sources[0] == {"book_title": "Book", "book_id": "b", "chapter": "Chapter 1",
                          "headings": ["Chapter 1", "Intro"], "pages": [10]}


def test_ask_ollama_sends_limits_and_parses_reply():
    sent = {}

    class Resp:
        def raise_for_status(self): pass
        def json(self):
            return {"message": {"content": " hello "}, "prompt_eval_count": 7,
                    "eval_count": 3, "done_reason": "stop"}

    def fake_post(url, json, timeout):
        sent.update(url=url, json=json, timeout=timeout)
        return Resp()

    out = qa.ask_ollama(CFG, [{"role": "user", "content": "q"}], post=fake_post)
    assert sent["url"] == "http://x:1/api/chat"
    assert sent["json"]["options"] == {"num_predict": 50, "num_ctx": 1000}
    assert sent["json"]["think"] is False and sent["json"]["stream"] is False
    assert out == {"text": "hello", "prompt_tokens": 7, "answer_tokens": 3, "done_reason": "stop"}
