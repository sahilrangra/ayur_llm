from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv
import os
os.environ["ANONYMIZED_TELEMETRY"] = "0"

import chromadb
from chromadb.config import Settings
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "ayurveda_docs"

load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
EMBED_MODEL = os.getenv("EMBEDDINGS_MODEL_NAME", "text-embedding-3-small").strip()

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing in .env")

client = OpenAI(api_key=OPENAI_API_KEY)


def embed_query(q: str) -> List[float]:
    resp = client.embeddings.create(model=EMBED_MODEL, input=[q])
    return resp.data[0].embedding


def format_hit(i: int, doc: str, meta: Dict[str, Any], dist: float) -> str:
    sim = 1.0 - float(dist)
    return (
        f"\n--- HIT {i} | sim≈{sim:.4f} ---\n"
        f"{meta.get('source')} | {meta.get('title')} | {meta.get('file_name')}\n"
        f"pages {meta.get('page_start')}-{meta.get('page_end')} | section: {meta.get('section')}\n"
        f"chunk_id: {meta.get('doc_id')} | tags: {meta.get('tags')}\n"
        f"{doc}\n"
    )


def main() -> None:
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(f"Missing Chroma DB: {CHROMA_DIR}. Run build first.")

    chroma = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    col = chroma.get_collection(COLLECTION_NAME)

    print("✅ Ready. Ask questions. Blank to exit.\n")
    while True:
        q = input("Q> ").strip()
        if not q:
            break

        q_emb = embed_query(q)
        res = col.query(
            query_embeddings=[q_emb],
            n_results=10,
            include=["documents", "metadatas", "distances"],
        )

        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]

        print("\n================ RESULTS ================\n")
        for i, (d, m, dist) in enumerate(zip(docs, metas, dists), start=1):
            print(format_hit(i, d, m, dist))
        print("========================================\n")


if __name__ == "__main__":
    main()
