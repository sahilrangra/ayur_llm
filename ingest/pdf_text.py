from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import re

import fitz

WHITESPACE_RE = re.compile(r"[ \t]+")
LINEBREAKS_RE = re.compile(r"\n{3,}")

@dataclass
class PageText:
    page_num: int  # 1-based
    text: str

@dataclass
class PdfExtractResult:
    path: Path
    title_guess: str
    pages: List[PageText]
    meta: Dict[str, str]
    extraction_notes: List[str]

def _clean_text(text: str) -> str:
    # normalize whitespace but keep line breaks (useful for headings)
    text = text.replace("\r", "\n")
    text = WHITESPACE_RE.sub(" ", text)
    text = LINEBREAKS_RE.sub("\n\n", text)
    # remove isolated hyphen line breaks like "man-\nagement"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return text.strip()

def extract_pdf_text(path: Path) -> PdfExtractResult:
    doc = fitz.open(path)
    meta = {}
    notes: List[str] = []
    try:
        m = doc.metadata or {}
        for k, v in m.items():
            if v:
                meta[str(k)] = str(v)
    except Exception as e:
        notes.append(f"metadata_error: {e}")

    pages: List[PageText] = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        text = _clean_text(text)
        pages.append(PageText(page_num=i + 1, text=text))

    # title guess: metadata title > first non-empty line of page 1
    title_guess = meta.get("title", "").strip()
    if not title_guess:
        first = pages[0].text.splitlines() if pages else []
        title_guess = next((ln.strip() for ln in first if ln.strip()), path.stem)

    return PdfExtractResult(
        path=path,
        title_guess=title_guess,
        pages=pages,
        meta=meta,
        extraction_notes=notes,
    )
