import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Ayurveda AI",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 Ayurveda AI")
st.write("Ask questions **only from your documents** (RAG + citations).")
st.markdown("⚡ Fast • Safe • Cited")

# -----------------------------
# Fake local backend (TEMP)
# -----------------------------
@st.cache_data(ttl=60)
def list_docs():
    return [
        {
            "doc_id": "charaka_1",
            "display_name": "Charaka Samhita – Sutrasthana",
            "source": "Charaka",
        },
        {
            "doc_id": "ashtanga_1",
            "display_name": "Ashtanga Hridayam – Sutrasthana",
            "source": "Ashtanga",
        },
    ]


def ask_question(payload):
    return {
        "answer": f"(Demo answer)\n\nYou asked:\n{payload['question']}",
        "retrieved_count": 2,
        "citations": [
            {
                "title": "Charaka Samhita",
                "source": "Charaka",
                "file_name": "charaka_samhita.pdf",
                "page_start": 12,
                "page_end": 13,
                "section": "Sutrasthana",
                "chunk_id": "chunk_001",
            }
        ],
    }

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ Controls")

docs = list_docs()
sources = sorted({d["source"] for d in docs})
source_choice = st.sidebar.selectbox(
    "Filter by Source (optional)",
    ["ALL"] + sources,
)

doc_label_to_id = {d["display_name"]: d["doc_id"] for d in docs}

selected_doc_labels = st.sidebar.multiselect(
    "Choose documents (optional)",
    options=list(doc_label_to_id.keys()),
)

top_k = st.sidebar.slider("Top K chunks", 3, 20, 8)
strict = st.sidebar.toggle("Strict mode (no guessing)", value=True)

# -----------------------------
# Main UI
# -----------------------------
question = st.text_input(
    "Ask your question",
    placeholder="e.g., What does Charaka say about healthy living?",
)

ask = st.button("✨ Ask", use_container_width=True)

if ask:
    if not question.strip():
        st.warning("Type a question first.")
        st.stop()

    payload = {
        "question": question.strip(),
        "top_k": top_k,
        "strict": strict,
        "doc_ids": [doc_label_to_id[x] for x in selected_doc_labels],
        "source_filter": None if source_choice == "ALL" else source_choice,
    }

    with st.spinner("Thinking with citations..."):
        data = ask_question(payload)

    st.subheader("✅ Answer")
    st.write(data["answer"])

    st.caption(f"Retrieved chunks: {data['retrieved_count']}")

    if data.get("citations"):
        st.subheader("📌 Citations")
        for c in data["citations"]:
            st.markdown(
                f"""
**{c['title']}**

- Source: {c['source']}
- File: {c['file_name']}
- Pages: {c['page_start']}–{c['page_end']}
- Section: {c['section']}
- Chunk: {c['chunk_id']}
"""
            )
