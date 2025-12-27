from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Iterable, List

from dotenv import load_dotenv
from tqdm import tqdm
import os
os.environ["ANONYMIZED_TELEMETRY"] = "0"

import chromadb
from chromadb.config import Settings

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JSONL_DIR = PROJECT_ROOT / "out" / "jsonl_clean"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "ayurveda_docs"

load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
EMBED_MODEL = os.getenv("EMBEDDINGS_MODEL_NAME", "text-embedding-3-small").strip()

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing in .env")

client = OpenAI(api_key=OPENAI_API_KEY)


def iter_jsonl_files() -> List[Path]:
    files = sorted(JSONL_DIR.glob("*_clean.jsonl"))
    if not files:
        raise FileNotFoundError(f"No *_clean.jsonl found in {JSONL_DIR}")
    return files


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def make_metadata(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "doc_id": rec.get("doc_id"),
        "file_name": rec.get("file_name"),
        "title": rec.get("title"),
        "source": rec.get("source"),
        "page_start": int(rec.get("page_start", 0) or 0),
        "page_end": int(rec.get("page_end", 0) or 0),
        "section": " > ".join(rec.get("section_path") or []),
        "tags": ",".join(rec.get("tags") or []),
    }


def embed_texts(texts: List[str], max_retries: int = 6) -> List[List[float]]:
    """
    Robust embeddings with retry/backoff.
    """
    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(
                model=EMBED_MODEL,
                input=texts,
            )
            return [d.embedding for d in resp.data]
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 20.0)


def batch(lst: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def main() -> None:
    if not JSONL_DIR.exists():
        raise FileNotFoundError(f"Missing: {JSONL_DIR}")

    files = iter_jsonl_files()
    print(f"✅ JSONL_DIR: {JSONL_DIR}")
    print(f"✅ CHROMA_DIR: {CHROMA_DIR}")
    print(f"✅ COLLECTION: {COLLECTION_NAME}")
    print(f"✅ OPENAI EMBED MODEL: {EMBED_MODEL}")

    chroma = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )

    # Clean rebuild (recommended for first stable index)
    existing = [c.name for c in chroma.list_collections()]
    if COLLECTION_NAME in existing:
        chroma.delete_collection(COLLECTION_NAME)

    col = chroma.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    for fp in files:
        for rec in load_jsonl(fp):
            txt = (rec.get("text") or "").strip()
            cid = (rec.get("chunk_id") or "").strip()
            if not txt or not cid:
                continue
            ids.append(cid)
            docs.append(txt)
            metas.append(make_metadata(rec))

    print(f"📦 Total chunks: {len(ids)}")

    BATCH = 96  # safe batch size; can bump later
    for idxs in tqdm(list(batch(list(range(len(ids))), BATCH)), desc="Embedding+Indexing"):
        b_ids = [ids[i] for i in idxs]
        b_docs = [docs[i] for i in idxs]
        b_meta = [metas[i] for i in idxs]

        embs = embed_texts(b_docs)

        col.add(
            ids=b_ids,
            documents=b_docs,
            metadatas=b_meta,
            embeddings=embs,
        )

    print("✅ DONE.")
    print(f"✅ Indexed chunks: {col.count()}")
    print("👉 Next: python vectordb/query_chroma_openai.py")


if __name__ == "__main__":
    main()
