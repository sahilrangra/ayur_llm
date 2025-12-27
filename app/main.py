from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
EMBED_MODEL = os.getenv("EMBEDDINGS_MODEL_NAME", "text-embedding-3-small").strip()

CHROMA_DIR = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "ayurveda_docs"

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing in .env")

oaiclient = OpenAI(api_key=OPENAI_API_KEY)

# Hard disable telemetry noise
os.environ["ANONYMIZED_TELEMETRY"] = "0"

chroma = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(anonymized_telemetry=False),
)
col = chroma.get_collection(COLLECTION_NAME)

print("✅ CHROMA_DIR =", CHROMA_DIR)
print("✅ Existing collections =", [c.name for c in chroma.list_collections()])
print("✅ Using collection =", COLLECTION_NAME)
print("✅ Collection count =", col.count())

# ✅ IMPORTANT: app must exist BEFORE any @app.get/@app.post
app = FastAPI(title="Ayurveda RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only; lock down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def pretty_doc_label(meta: Dict[str, Any]) -> str:
    # Smart “actual name” label: Title + Source + File
    title = (meta.get("title") or "").strip()
    source = (meta.get("source") or "").strip()
    file_name = (meta.get("file_name") or "").strip()

    # Fallbacks
    if not title:
        title = file_name or "Untitled"
    if not file_name:
        file_name = "unknown.pdf"
    if not source:
        source = "UNKNOWN"

    # Beautiful label
    # Example: "Charaka Samhita  •  CLASSICAL  •  Charak_Samhita.pdf"
    return f"{title}  •  {source}  •  {file_name}"

class AskRequest(BaseModel):
    question: str = Field(..., min_length=2)
    top_k: int = 8
    doc_ids: Optional[List[str]] = None   # <-- allow selecting 2–3 docs
    source_filter: Optional[str] = None  # WHO / CLASSICAL / AYUSH/GOV / CCRAS
    strict: bool = True


class Citation(BaseModel):
    source: str
    title: str
    file_name: str
    page_start: int
    page_end: int
    section: str
    tags: str
    chunk_id: str


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    retrieved_count: int


def embed_query(q: str) -> List[float]:
    resp = oaiclient.embeddings.create(model=EMBED_MODEL, input=[q])
    return resp.data[0].embedding


def retrieve(
    question: str,
    top_k: int,
    source_filter: Optional[str],
    doc_ids: Optional[List[str]],
) -> List[Dict[str, Any]]:

    q_emb = embed_query(question)

    where: Dict[str, Any] = {}

    # 🔹 Filter by source (WHO / CLASSICAL / etc.)
    if source_filter:
        where["source"] = source_filter

    # 🔹 Filter by specific documents
    if doc_ids:
        where["doc_id"] = {"$in": doc_ids}

    res = col.query(
        query_embeddings=[q_emb],
        n_results=int(top_k),
        include=["documents", "metadatas", "distances"],
        where=where if where else None,  # IMPORTANT
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out: List[Dict[str, Any]] = []
    for d, m, dist in zip(docs, metas, dists):
        if not d or not str(d).strip():
            continue
        out.append({"text": d, "meta": m or {}, "distance": dist})

    return out


def build_prompt(question: str, contexts: List[Dict[str, Any]], strict: bool) -> str:
    rules = [
        "You are an Ayurveda assistant. Answer ONLY using the provided sources.",
        "If the sources do not contain the answer, say: 'I don’t have enough information in the provided documents to answer this.'",
        "Do NOT invent facts, dosages, or medical claims.",
        "Be concise and practical.",
    ]
    if not strict:
        rules[1] = "If sources are weak, answer cautiously and explicitly say what is missing."

    ctx_lines = []
    for i, c in enumerate(contexts, start=1):
        m = c["meta"]
        header = (
            f"[SOURCE {i}] source={m.get('source')} | title={m.get('title')} | "
            f"file={m.get('file_name')} | pages={m.get('page_start')}-{m.get('page_end')} | "
            f"section={m.get('section')}"
        )
        ctx_lines.append(header + "\n" + c["text"])

    return (
        "RULES:\n- " + "\n- ".join(rules) + "\n\n"
        "SOURCES:\n" + "\n\n".join(ctx_lines) + "\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER (with short citations like [SOURCE 2], [SOURCE 4]):"
    )


@app.get("/debug_query")
def debug_query(
    q: str = Query(..., min_length=2),
    top_k: int = 5,
    source: str | None = None,
):
    emb = embed_query(q)

    kwargs: Dict[str, Any] = {}
    if source:
        kwargs["where"] = {"source": source}

    res = col.query(
        query_embeddings=[emb],
        n_results=int(top_k),
        include=["documents", "metadatas", "distances"],
        **kwargs,
    )

    docs0 = (res.get("documents") or [[]])[0]
    metas0 = (res.get("metadatas") or [[]])[0]
    dists0 = (res.get("distances") or [[]])[0]

    return {
        "collection_count": col.count(),
        "query_len": len(q),
        "emb_dim": len(emb),
        "top_k": top_k,
        "source_filter": source,
        "res_keys": list(res.keys()),
        "docs_len": len(docs0),
        "metas_len": len(metas0),
        "dists_len": len(dists0),
        "first_distance": (dists0[0] if dists0 else None),
        "first_meta": (metas0[0] if metas0 else None),
        "first_doc_preview": ((docs0[0] or "")[:250] if docs0 else None),
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    contexts = retrieve(
        question=req.question,
        top_k=req.top_k,
        source_filter=req.source_filter,
        doc_ids=req.doc_ids,
    )

    if not contexts:
        return AskResponse(
            answer="I don’t have enough information in the provided documents to answer this.",
            citations=[],
            retrieved_count=0,
        )

    prompt = build_prompt(req.question, contexts, req.strict)

    completion = oaiclient.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a careful RAG assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    answer = completion.choices[0].message.content.strip()

    citations: List[Citation] = []
    for c in contexts:
        m = c["meta"] or {}
        citations.append(
            Citation(
                source=str(m.get("source") or ""),
                title=str(m.get("title") or ""),
                file_name=str(m.get("file_name") or ""),
                page_start=int(m.get("page_start") or 0),
                page_end=int(m.get("page_end") or 0),
                section=str(m.get("section") or ""),
                tags=str(m.get("tags") or ""),
                chunk_id=str(m.get("chunk_id") or "")
            )
        )

    return AskResponse(answer=answer, citations=citations, retrieved_count=len(contexts))


@app.get("/health")
def health():
    return {
        "chroma_dir": str(CHROMA_DIR),
        "collections": [c.name for c in chroma.list_collections()],
        "collection": COLLECTION_NAME,
        "count": col.count(),
        "embed_model": EMBED_MODEL,
        "chat_model": OPENAI_MODEL,
    }


@app.get("/peek")
def peek():
    got = col.get(limit=1, include=["documents", "metadatas"])
    return {
        "count": col.count(),
        "sample_ids": got.get("ids", []),
        "sample_meta": (got.get("metadatas") or [None])[0],
        "sample_doc_preview": ((got.get("documents") or [""])[0] or "")[:300],
    }

@app.get("/list_docs")
def list_docs():
    """
    Returns unique documents with a beautiful display name.
    Uses Chroma metadata.
    """
    # Pull all metadatas (fast enough for ~1k chunks)
    got = col.get(include=["metadatas"], limit=col.count())
    metas = got.get("metadatas") or []

    by_doc: Dict[str, Dict[str, Any]] = {}
    for m in metas:
        if not m:
            continue
        doc_id = str(m.get("doc_id") or "").strip()
        if not doc_id:
            continue
        # Keep first seen (good enough)
        if doc_id not in by_doc:
            by_doc[doc_id] = m

    docs = []
    for doc_id, m in by_doc.items():
        docs.append(
            {
                "doc_id": doc_id,
                "title": str(m.get("title") or ""),
                "source": str(m.get("source") or ""),
                "file_name": str(m.get("file_name") or ""),
                "display_name": pretty_doc_label(m),
            }
        )

    # Sort nicely by source then title
    docs.sort(key=lambda x: (x["source"], x["title"], x["file_name"]))
    return {"count": len(docs), "docs": docs}
