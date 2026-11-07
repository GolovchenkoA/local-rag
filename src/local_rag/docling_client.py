"""HTTP calls to the docling-serve container (only the parse and chunk stages need it)."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import httpx

log = logging.getLogger(__name__)


def _b(value: bool) -> str:
    return "true" if value else "false"


def _convert_fields(cfg: SimpleNamespace, prefix: str = "") -> list[tuple[str, str]]:
    c = cfg.docling.convert
    fields = [
        (f"{prefix}do_ocr", _b(c.do_ocr)),
        (f"{prefix}include_images", _b(c.include_images)),
        (f"{prefix}do_picture_description", _b(c.do_picture_description)),
        (f"{prefix}do_pdf_heading_hierarchy", _b(c.do_pdf_heading_hierarchy)),
    ]
    if c.do_ocr:
        fields.append((f"{prefix}ocr_engine", c.ocr_engine))
        fields += [(f"{prefix}ocr_lang", lang) for lang in c.ocr_languages]
    return fields


def _post(cfg: SimpleNamespace, endpoint: str, pdf: Path, fields: list[tuple[str, str]]) -> dict:
    url = cfg.docling.url.rstrip("/") + endpoint
    log.info("POST %s (%s, %.1f MB)", url, pdf.name, pdf.stat().st_size / 1e6)
    with open(pdf, "rb") as f:
        files = {"files": (pdf.name, f, "application/pdf")}
        resp = httpx.post(url, data=fields, files=files,
                          timeout=cfg.docling.timeout_seconds)
    resp.raise_for_status()
    return resp.json()


def convert_pdf(cfg: SimpleNamespace, pdf: Path) -> dict:
    """POST /v1/convert/file. Response: {'document': {'json_content', 'md_content'}, ...}."""
    fields = [("to_formats", "json"), ("to_formats", "md")] + _convert_fields(cfg)
    return _post(cfg, "/v1/convert/file", pdf, fields)


def chunk_pdf(cfg: SimpleNamespace, pdf: Path) -> dict:
    """POST /v1/chunk/hybrid/file. Response: {'chunks': [...], ...}."""
    ch = cfg.chunking
    fields = _convert_fields(cfg, prefix="convert_") + [
        ("chunking_tokenizer", ch.tokenizer),
        ("chunking_max_tokens", str(ch.max_tokens)),
        ("chunking_merge_peers", _b(ch.merge_peers)),
        ("chunking_include_raw_text", "true"),
        ("include_converted_doc", "false"),
    ]
    return _post(cfg, "/v1/chunk/hybrid/file", pdf, fields)
