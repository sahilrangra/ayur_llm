# FILE: ingest/heuristics.py
from __future__ import annotations

import re
from typing import List

# Strong heading patterns (numbered sections and ALL CAPS report headers)
NUM_HEADING_RE = re.compile(r"^\s*(\d+(\.\d+){0,6})\s+[A-Za-z].+")
ALLCAPS_RE = re.compile(r"^[A-Z][A-Z0-9 \-&,()/]{6,}$")

# Lines that must NEVER be treated as headings
BAD_HEADING_RE = re.compile(
    r"("
    r"\bpage\b\s*[ivxlcdm0-9]+\b|"                 # Page ii / Page 10
    r"\bof\s+\d+\b|"                               # "of 38"
    r"\bwww\.\b|https?://|@|"                      # urls/emails
    r"\bphone\b|\be-?mail\b|\bfax\b|"              # contact blocks
    r"\bministry\b.*\bhealth\b|"                   # ministry boilerplate often on cover
    r"\bgovernment\b.*\bindia\b|"                  # boilerplate
    r"\bdepartment\s+of\s+ayush\b|"                # boilerplate
    r"\bsecretary\b|"                              # roles
    r"\bdg,?\s*ccras\b|\bddg,?\s*ccras\b|"         # people/designations
    r"\bcompiled\s+by\b|\breviewer\b|\beditor\b|\bpublisher\b|"
    r"\bsupervision\b|\bguidance\b|\bcompiled\b|"  # cover roles
    r"\bforeword\b$|\bcontents\b$|\bpreface\b$|"   # handled via whitelist
    r"\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b|"  # dates
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b\s+\d{4}\b|"
    r"\bcopyright\b|\bisbn\b|\ball\s+rights\s+reserved\b|"  # legal boilerplate
    r"\bprinted\s+by\b|\bprinter\b|\bwebsite\b|\be-mail\b"
    r")",
    re.IGNORECASE,
)

# Allowed short headings (front matter)
FRONT_MATTER = {
    "preface",
    "contents",
    "foreword",
    "introduction",
    "glossary",
    "acknowledgements",
    "acknowledgments",
    "index",
    "prologue",
    "executive summary",
}

# Sentence-ish / narrative verbs that often appear in non-heading lines
SENTENCE_VERB_RE = re.compile(
    r"\b(is|are|was|were|have|has|had|will|shall|should|can|could|may|might)\b",
    re.IGNORECASE,
)


def infer_source(title: str) -> str:
    """
    Lightweight pre-classification based on title only.
    postprocess_json will do final, more accurate classification.
    """
    t = (title or "").lower()
    if "who" in t or "world health organization" in t:
        return "WHO"
    if "ayush" in t or "npcdcs" in t:
        return "AYUSH/GOV"
    if "ccras" in t or "central council for research in ayurved" in t:
        return "CCRAS"
    if "charaka" in t or "caraka" in t:
        return "CLASSICAL"
    return "UNKNOWN"


def auto_tags(title: str) -> List[str]:
    t = (title or "").lower()
    tags: List[str] = []
    if "safety" in t or "benchmark" in t:
        tags += ["safety", "regulation"]
    if "diet" in t or "lifestyle" in t:
        tags += ["diet", "lifestyle"]
    if "strategy" in t or "policy" in t:
        tags += ["policy", "evidence"]
    if "charaka" in t or "caraka" in t:
        tags += ["classical", "sutra"]

    # unique preserve order
    out: List[str] = []
    for x in tags:
        if x not in out:
            out.append(x)
    return out


def looks_like_heading(line: str, min_len: int, max_len: int) -> bool:
    """
    GOAT heading detector:
    - blocks watermarks, addresses, roles, dates, page labels
    - accepts:
      (a) numbered headings: "6.3 Strategic objective 3..."
      (b) ALL CAPS headings: "CONTENTS", "INTRODUCTION"
      (c) whitelisted short headings: "Preface", "Foreword", etc.
    """
    s = (line or "").strip()
    if len(s) < min_len or len(s) > max_len:
        return False

    low = s.lower()

    # Allow simple front matter headings exactly
    if low in FRONT_MATTER:
        return True

    # Reject obvious bad heading candidates
    if BAD_HEADING_RE.search(s):
        return False

    # Reject if it looks like a normal sentence (GOAT filter)
    # Too many commas often means it's a clause/sentence
    if s.count(",") >= 2 and not s.isupper():
        return False

    # Common verb words => usually sentence, not a heading
    if SENTENCE_VERB_RE.search(low):
        return False

    # Period-ending lines are almost never headings (except ALL CAPS)
    if s.endswith(".") and not s.isupper():
        return False

    # Colon-ending long lines are often "sentence labels" not headings
    if s.endswith(":") and len(s.split()) > 8 and not s.isupper():
        return False

    # Comma-ending lines are almost never headings (except ALL CAPS)
    if s.endswith(",") and not s.isupper():
        return False

    # Strong patterns
    if NUM_HEADING_RE.match(s):
        return True

    if ALLCAPS_RE.match(s):
        return True

    # Otherwise, be conservative
    return False
