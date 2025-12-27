# FILE: ingest/chunking.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import re


@dataclass
class Chunk:
    page_start: int
    page_end: int
    section_path: List[str]
    text: str


# ---- Front-matter boundaries (the biggest "RAG killer") ----
FRONT_SPLIT_MARKERS = [
    "table of contents",
    "contents",
    "index",
    "preface",
    "foreword",
    "introduction",
    "prologue",
    "executive summary",
    "glossary",
    "acknowledgements",
    "acknowledgments",
]

FRONT_SPLIT_RE = re.compile(
    r"^\s*("
    + "|".join(re.escape(m) for m in FRONT_SPLIT_MARKERS)
    + r")\s*$",
    re.IGNORECASE,
)

# Also split if line starts with "CONTENTS" and then a bunch of dotted leaders etc
TOC_SIGNAL_RE = re.compile(r"^\s*(contents|table of contents)\s*$", re.IGNORECASE)
TOC_LEADER_RE = re.compile(r"\.{3,}|(\bpage\b\s*\d+)", re.IGNORECASE)

# Remove very common scanned page noise lines
NOISE_RE = re.compile(
    r"^\s*(page\s*[ivxlcdm0-9]+(\s*of\s*\d+)?)\s*$",
    re.IGNORECASE,
)


def _dedupe_adjacent(seq: List[str]) -> List[str]:
    out: List[str] = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


def _clean_lines(text: str) -> List[str]:
    lines = []
    for raw in (text or "").splitlines():
        ln = (raw or "").strip()
        if not ln:
            continue
        if NOISE_RE.match(ln):
            continue
        lines.append(ln)
    return lines


def _split_page_into_blocks(lines: List[str]) -> List[List[str]]:
    """
    Split within a page when we detect strong front-matter markers.
    This is the key to keeping "Contents" from polluting Preface chunks.
    """
    if not lines:
        return []

    blocks: List[List[str]] = []
    cur: List[str] = []

    # Detect if page looks like a TOC page
    tocish = False
    leaders = 0
    for ln in lines[:80]:
        if TOC_SIGNAL_RE.match(ln):
            tocish = True
        if TOC_LEADER_RE.search(ln):
            leaders += 1
    if leaders >= 8:
        tocish = True

    for ln in lines:
        # Hard split at front markers
        if FRONT_SPLIT_RE.match(ln):
            if cur:
                blocks.append(cur)
                cur = []
            blocks.append([ln])  # marker becomes its own tiny block header
            continue

        # If it's a TOC-ish page, and we see "Preface" etc, split there
        if tocish and ln.lower() in FRONT_SPLIT_MARKERS:
            if cur:
                blocks.append(cur)
                cur = []
            blocks.append([ln])
            continue

        cur.append(ln)

    if cur:
        blocks.append(cur)

    return blocks


def build_chunks_from_pages(
    pages: List[Tuple[int, str]],
    target_chars: int,
    overlap_chars: int,
    section_paths_by_page: Dict[int, List[str]],
) -> List[Chunk]:
    """
    Production-grade chunker:
    - cleans noise
    - splits pages into semantic blocks (front matter boundaries)
    - builds chunks roughly target_chars, respecting block boundaries
    - merges small blocks but never merges across a front-matter marker
    - carries stable section_path by page
    """
    chunks: List[Chunk] = []

    cur_text_parts: List[str] = []
    cur_start: Optional[int] = None
    cur_end: Optional[int] = None
    cur_section: List[str] = ["Document"]

    def flush():
        nonlocal cur_text_parts, cur_start, cur_end, cur_section
        if not cur_text_parts or cur_start is None or cur_end is None:
            return
        txt = "\n".join(cur_text_parts).strip()
        if not txt:
            cur_text_parts, cur_start, cur_end = [], None, None
            cur_section = ["Document"]
            return
        chunks.append(
            Chunk(
                page_start=cur_start,
                page_end=cur_end,
                section_path=_dedupe_adjacent(cur_section),
                text=txt,
            )
        )
        cur_text_parts, cur_start, cur_end = [], None, None
        cur_section = ["Document"]

    def start_new(pno: int):
        nonlocal cur_start, cur_end, cur_section
        cur_start = pno
        cur_end = pno
        cur_section = section_paths_by_page.get(pno, ["Document"])

    def apply_overlap(prev_text: str):
        # add overlap from previous chunk tail
        if overlap_chars <= 0:
            return []
        tail = prev_text[-overlap_chars:]
        return [tail] if tail.strip() else []

    for (pno, raw_text) in pages:
        page_section = section_paths_by_page.get(pno, ["Document"])
        page_section = _dedupe_adjacent(page_section)

        lines = _clean_lines(raw_text)
        blocks = _split_page_into_blocks(lines)

        # Convert blocks to text blobs
        block_texts: List[str] = []
        for b in blocks:
            bt = "\n".join(b).strip()
            if bt:
                block_texts.append(bt)

        for bt in block_texts:
            bt_low = bt.strip().lower()

            # If this is a front marker alone ("Preface", "Contents"), it should start a fresh chunk.
            is_front_marker = FRONT_SPLIT_RE.match(bt.strip()) is not None and len(bt.splitlines()) == 1

            # If we’re starting fresh
            if cur_start is None:
                start_new(pno)

            # If section changes across pages, we may want to flush (keeps sections clean)
            # Only enforce when page changes and we already have content.
            if cur_text_parts and cur_end != pno:
                if page_section != _dedupe_adjacent(cur_section):
                    flush()
                    start_new(pno)

            # If the next block is a front marker, flush current first.
            if is_front_marker:
                if cur_text_parts:
                    flush()
                start_new(pno)
                cur_text_parts.append(bt)  # marker itself
                cur_end = pno
                flush()
                continue

            # If adding this block exceeds size, flush before adding
            current_len = sum(len(x) for x in cur_text_parts) + max(0, len(cur_text_parts) - 1)
            if cur_text_parts and current_len + 1 + len(bt) > target_chars:
                # flush and carry overlap
                prev_txt = "\n".join(cur_text_parts)
                flush()
                start_new(pno)
                cur_text_parts.extend(apply_overlap(prev_txt))

            # Add block
            cur_text_parts.append(bt)
            cur_end = pno
            cur_section = page_section  # always keep latest page section for the chunk

    # final flush
    flush()
    return chunks
