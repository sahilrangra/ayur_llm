from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any

# Remove ASCII control chars + C1 control chars (the \x82 bullet issue lives here)
CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")

# Collapse multi-space / tabs
MULTISPACE_RE = re.compile(r"[ \t]{2,}")

BAD_TITLE_MARKERS = (
    "new doc", "newstarting", "microsoft word", ".doc", "starting content pages"
)

# Common broken bullets coming from PDF text extraction
# \u0082 = the “\x82” bullet you saw
BULLET_NORMALIZE = {
    "\u0082": "• ",   # C1 “control” bullet (should be removed by CTRL_RE, but normalize anyway)
    "\u0095": "• ",
    "": "• ",
    "◦": "• ",
    "▪": "• ",
    "□": "• ",        # keep list structure for some OCR docs
}

def clean_str(s: str) -> str:
    if not s:
        return ""
    # 1) normalize common bullets BEFORE stripping controls (just in case)
    for k, v in BULLET_NORMALIZE.items():
        s = s.replace(k, v)

    # 2) strip control chars including C1 range
    s = CTRL_RE.sub("", s)

    # 3) normalize whitespace
    s = MULTISPACE_RE.sub(" ", s)

    # 4) normalize weird line-only bullet artifacts
    s = re.sub(r"\n[•]\s*\n", "\n", s)

    return s.strip()

def is_bad_title(title: str) -> bool:
    t = (title or "").strip().lower()
    if not t:
        return True
    return any(m in t for m in BAD_TITLE_MARKERS)

def infer_source_from(filename: str, title: str, sample_text: str) -> str:
    f = (filename or "").lower()
    t = (title or "").lower()
    x = (sample_text or "").lower()

    # WHO
    if ("who" in f) or ("who " in t) or ("world health organization" in x) or ("who benchmarks" in x):
        return "WHO"

    # AYUSH / Govt
    if ("npcdcs" in f) or ("integration of ayush" in x) or ("ministry of ayush" in x) or ("government of india" in x and "ayush" in x):
        return "AYUSH/GOV"

    # CCRAS
    if ("ccras" in x) or ("central council for research in ayurveda" in x) or ("research in ayurveda and siddha" in x):
        return "CCRAS"

    # Classical
    if ("charaka" in f) or ("caraka" in f) or ("charaka samhita" in x):
        return "CLASSICAL"

    # Dossier (keep as CCRAS if your project wants it grouped that way)
    if ("science_of_lifedossier" in f) or ("the science of life" in x):
        return "CCRAS"

    return "UNKNOWN"

def choose_better_title(filename: str, old_title: str, sample_text: str) -> str:
    if not is_bad_title(old_title):
        return clean_str(old_title)

    lines = [clean_str(l) for l in (sample_text or "").splitlines()]
    lines = [l for l in lines if l and len(l) >= 6]

    for l in lines[:30]:
        if 8 < len(l) < 120:
            return l

    return Path(filename).stem.replace("_", " ").strip()

def postprocess_one(in_path: Path, out_path: Path) -> Dict[str, Any]:
    count = 0
    sample_text = ""
    old_title = ""
    filename = in_path.name

    # sample from first chunk
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            old_title = rec.get("title", "") or ""
            sample_text = rec.get("text", "") or ""
            break

    fixed_title = choose_better_title(filename, old_title, sample_text)
    fixed_source = infer_source_from(filename, fixed_title, sample_text)

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)

            rec["title"] = fixed_title
            rec["source"] = fixed_source

            # clean text
            rec["text"] = clean_str(rec.get("text", ""))

            # clean section_path
            sp = rec.get("section_path", [])
            if isinstance(sp, list):
                cleaned = [clean_str(str(x)) for x in sp]
                cleaned = [x for x in cleaned if x]
                # de-dup adjacent
                dedup = []
                for x in cleaned:
                    if not dedup or dedup[-1] != x:
                        dedup.append(x)
                rec["section_path"] = dedup if dedup else ["Document"]
            else:
                sps = clean_str(str(sp)) if sp else ""
                rec["section_path"] = [sps] if sps else ["Document"]

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1

    return {"in": str(in_path), "out": str(out_path), "chunks": count, "source": fixed_source, "title": fixed_title}

def main() -> None:
    in_dir = Path("out/jsonl")
    out_dir = Path("out/jsonl_clean")
    out_dir.mkdir(parents=True, exist_ok=True)
    (Path("out/manifests")).mkdir(parents=True, exist_ok=True)

    reports = []
    for p in sorted(in_dir.glob("*.jsonl")):
        out_p = out_dir / p.name.replace(".jsonl", "_clean.jsonl")
        reports.append(postprocess_one(p, out_p))

    (Path("out/manifests") / "postprocess_report.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("✅ Clean JSONLs written to out/jsonl_clean/")
    print("✅ Report: out/manifests/postprocess_report.json")

if __name__ == "__main__":
    main()
