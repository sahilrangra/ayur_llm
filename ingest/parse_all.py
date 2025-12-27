# FILE: ingest/parse_all.py
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

from tqdm import tqdm

from ingest.config import ParseConfig
from ingest.pdf_text import extract_pdf_text
from ingest.heuristics import looks_like_heading, infer_source, auto_tags
from ingest.chunking import build_chunks_from_pages


def _doc_id_from_path(path: Path) -> str:
    h = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in path.stem).strip("_")
    return f"{slug}_{h}"


def _dedupe_adjacent(seq: List[str]) -> List[str]:
    out: List[str] = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


def _build_section_paths_strict_heading_only(
    pages_text: Dict[int, str],
    min_h: int,
    max_h: int,
) -> Dict[int, List[str]]:
    """
    GOAT MODE section labeling:
    - no TOC page->page mapping
    - only accept headings via strict looks_like_heading()
    - stable depth up to 3: ["Document", H1] or ["Document", H1, H2]
    - dedupe adjacent repeats
    """
    section_by_page: Dict[int, List[str]] = {}
    current: List[str] = ["Document"]

    for pno in sorted(pages_text.keys()):
        text = pages_text[pno]

        heading_found: str | None = None
        for ln in text.splitlines():
            ln = (ln or "").strip()
            if not ln:
                continue
            if looks_like_heading(ln, min_h, max_h):
                heading_found = ln
                break

        if heading_found:
            # Decide whether it's H1 or H2:
            # - numbered headings or ALL CAPS are treated as H1 replacements
            is_numbered = heading_found[:1].isdigit()
            is_allcaps = heading_found.isupper()

            if is_numbered or is_allcaps or len(current) == 1:
                current = ["Document", heading_found]
            else:
                # treat as H2 under the current H1
                current = ["Document", current[1], heading_found]

            current = _dedupe_adjacent(current)

        section_by_page[pno] = current.copy()

    return section_by_page


def parse_all_pdfs(cfg: ParseConfig) -> None:
    in_dir = cfg.input_dir
    out_dir = cfg.out_dir
    (out_dir / "jsonl").mkdir(parents=True, exist_ok=True)
    (out_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    pdfs = sorted(list(in_dir.glob("*.pdf")))
    manifest: Dict[str, Any] = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "input_dir": str(in_dir),
        "doc_count": len(pdfs),
        "docs": [],
    }

    for pdf_path in tqdm(pdfs, desc="Parsing PDFs"):
        res = extract_pdf_text(pdf_path)
        doc_id = _doc_id_from_path(pdf_path)
        title = (res.title_guess or pdf_path.stem).strip()

        pages: List[Tuple[int, str]] = [
            (p.page_num, p.text)
            for p in res.pages
            if len((p.text or "").strip()) >= cfg.drop_tiny_pages_below_chars
        ]
        pages_text_map = {pno: txt for pno, txt in pages}

        section_by_page = _build_section_paths_strict_heading_only(
            pages_text=pages_text_map,
            min_h=cfg.min_heading_len,
            max_h=cfg.max_heading_len,
        )

        chunks = build_chunks_from_pages(
            pages=pages,
            target_chars=cfg.target_chars,
            overlap_chars=cfg.overlap_chars,
            section_paths_by_page=section_by_page,
        )

        # pre-tags; postprocess will refine
        source = infer_source(title)
        tags = auto_tags(title)

        out_jsonl = out_dir / "jsonl" / f"{doc_id}.jsonl"
        with out_jsonl.open("w", encoding="utf-8") as f:
            for i, ch in enumerate(chunks):
                rec = {
                    "chunk_id": f"{doc_id}::c{i:05d}",
                    "doc_id": doc_id,
                    "source": source,
                    "title": title,
                    "file_name": pdf_path.name,
                    "page_start": ch.page_start,
                    "page_end": ch.page_end,
                    "section_path": _dedupe_adjacent(ch.section_path if isinstance(ch.section_path, list) else ["Document"]),
                    "tags": tags,
                    "text": ch.text,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        manifest["docs"].append(
            {
                "doc_id": doc_id,
                "title": title,
                "source": source,
                "file_name": pdf_path.name,
                "page_count": len(res.pages),
                "kept_pages": len(pages),
                "chunk_count": len(chunks),
                "metadata": res.meta,
                "notes": res.extraction_notes,
                "jsonl_path": str(out_jsonl),
            }
        )

    (out_dir / "manifests" / "docs_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("✅ Done. JSONL written to out/jsonl and manifest to out/manifests/docs_manifest.json")


if __name__ == "__main__":
    cfg = ParseConfig(
        input_dir=Path("data/pdfs"),
        out_dir=Path("out"),
    )
    parse_all_pdfs(cfg)
