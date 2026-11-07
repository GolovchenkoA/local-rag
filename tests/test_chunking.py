from types import SimpleNamespace

from local_rag.chunking import build_chunks, read_jsonl, write_jsonl
from local_rag.ids import chunk_id

BOOK = SimpleNamespace(book_id="demo", book_title="Demo Book",
                       skip_top_headings=["Demo Book", "Index"])


def raw_chunk(text, headings, pages, tokens=100):
    return {"text": text, "raw_text": text, "headings": headings,
            "page_numbers": pages, "num_tokens": tokens}


RAW = {"chunks": [
    raw_chunk("1. Preface\n- a. Intro", None, [3, 4]),                   # table of contents
    raw_chunk("Demo Book\nby Someone", ["Demo Book"], [1]),              # title page
    raw_chunk("Chapter 1. Start\nWelcome", ["Chapter 1. Start"], [10]),
    raw_chunk("Chapter 1. Start\nTIP\nUse a venv.", ["Chapter 1. Start", "TIP"], [10, 11]),
    raw_chunk("Index\nabc", ["Index"], [99]),                            # index
]}


def test_filters_front_matter_and_index_but_keeps_callouts():
    chunks = build_chunks(RAW, BOOK, "v1")
    assert [c["headings"] for c in chunks] == [["Chapter 1. Start"], ["Chapter 1. Start", "TIP"]]


def test_position_is_contiguous_and_fields_are_filled():
    chunks = build_chunks(RAW, BOOK, "v1")
    assert [c["position"] for c in chunks] == [0, 1]
    c = chunks[1]
    assert c["book_id"] == "demo" and c["book_title"] == "Demo Book"
    assert c["chapter"] == "Chapter 1. Start"
    assert c["pages"] == [10, 11] and c["token_count"] == 100
    assert c["type"] == "chunk" and c["chunker_version"] == "v1"
    assert "TIP" in c["text"]


def test_ids_are_deterministic_and_unique():
    a = build_chunks(RAW, BOOK, "v1")
    b = build_chunks(RAW, BOOK, "v1")
    assert [c["id"] for c in a] == [c["id"] for c in b]
    assert len({c["id"] for c in a}) == len(a)
    assert a[0]["id"] == chunk_id("demo", ["Chapter 1. Start"], 0)


def test_jsonl_round_trip(tmp_path):
    chunks = build_chunks(RAW, BOOK, "v1")
    path = tmp_path / "sub" / "chunks.jsonl"
    write_jsonl(path, chunks)
    assert read_jsonl(path) == chunks
    assert len(path.read_text(encoding="utf-8").splitlines()) == len(chunks)
