import os
import streamlit as st
import chromadb
from chromadb.config import Settings
from openai import OpenAI

# =========================================================
# CONFIG (MUST MATCH YOUR INGESTION)
# =========================================================
CHROMA_PATH = "chroma_db"                 # folder already present
COLLECTION_NAME = "ayurveda_docs"              # ⚠️ change if your ingestion used another name
EMBED_MODEL = "text-embedding-3-small"    # must match ingestion
LLM_MODEL = "gpt-4.1-mini"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================================================
# STREAMLIT PAGE
# =========================================================
st.set_page_config(
    page_title="Ayurveda AI",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 Ayurveda AI")
st.write("Ask questions **only from your documents** (RAG + citations).")
st.markdown("⚡ Fast • Safe • Cited")

# =========================================================
# LOAD CHROMA DB (ONCE PER SESSION)
# =========================================================
@st.cache_resource
def load_chroma_collection():
    chroma_client = chromadb.Client(
        Settings(
            persist_directory=CHROMA_PATH,
            anonymized_telemetry=False,
        )
    )
    return chroma_client.get_collection(name=COLLECTION_NAME)

collection = load_chroma_collection()

# =========================================================
# SIDEBAR CONTROLS
# =========================================================
st.sidebar.header("⚙️ Controls")
top_k = st.sidebar.slider("Top K chunks", min_value=3, max_value=20, value=8)
strict = st.sidebar.toggle("Strict mode (no guessing)", value=True)

# =========================================================
# QUERY UI
# =========================================================
question = st.text_input(
    "Ask your question",
    placeholder="e.g., According to Charaka Samhita, what is dinacharya?",
)

ask = st.button("✨ Ask", use_container_width=True)

# =========================================================
# RETRIEVAL
# =========================================================
def retrieve_chunks(query: str, k: int):
    results = collection.query(
        query_texts=[query],
        n_results=k,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]

    chunks = []
    for text, meta, cid in zip(documents, metadatas, ids):
        chunks.append(
            {
                "text": text,
                "meta": meta or {},
                "chunk_id": cid,
            }
        )
    return chunks

# =========================================================
# GENERATION (STRICT RAG)
# =========================================================
def generate_answer(question: str, chunks: list, strict: bool):
    context = "\n\n".join(
        f"[{i+1}] {c['text']}" for i, c in enumerate(chunks)
    )

    system_prompt = (
        "You are an expert Ayurvedic scholar.\n"
        "You must answer ONLY using the provided context from Ayurvedic texts.\n"
    )
    if strict:
        system_prompt += (
            "If the answer is not explicitly stated in the context, "
            "reply exactly with: 'Not found in the provided Ayurvedic texts.'"
        )

    user_prompt = f"""
Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content

# =========================================================
# RUN PIPELINE
# =========================================================
if ask:
    if not question.strip():
        st.warning("Type a question first.")
        st.stop()

    with st.spinner("🔍 Searching Ayurvedic texts..."):
        chunks = retrieve_chunks(question, top_k)

    if not chunks:
        st.warning("No relevant passages found in the documents.")
        st.stop()

    with st.spinner("🧠 Generating answer from texts..."):
        answer = generate_answer(question, chunks, strict)

    # -------------------------
    # ANSWER
    # -------------------------
    st.subheader("✅ Answer")
    st.write(answer)
    st.caption(f"Retrieved chunks: {len(chunks)}")

    # -------------------------
    # CITATIONS
    # -------------------------
    st.subheader("📌 Citations")

    for c in chunks:
        meta = c["meta"]

        title = (
            meta.get("title")
            or meta.get("book")
            or meta.get("source")
            or "Ayurvedic Text"
        )

        source = meta.get("source") or meta.get("book") or ""
        file_name = meta.get("file_name") or meta.get("file") or ""
        section = meta.get("section") or meta.get("chapter") or ""
        page = meta.get("page") or meta.get("verse") or ""

        st.markdown(
            f"""
**{title}**

- Source: {source}
- File: {file_name}
- Section: {section}
- Page / Verse: {page}
- Chunk ID: `{c['chunk_id']}`
"""
        )
