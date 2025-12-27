from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ParseConfig:
    input_dir: Path
    out_dir: Path

    # chunking
    target_chars: int = 2200     # ~350-600 tokens depending on text
    overlap_chars: int = 250

    # heading heuristics
    min_heading_len: int = 4
    max_heading_len: int = 120

    # quality / safety
    drop_tiny_pages_below_chars: int = 30  # ignore near-empty pages (scans / blanks)
