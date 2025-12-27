import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Ayurveda AI",
    page_icon="🌿",
    layout="wide",
)

# ---- Sexy CSS ----
st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 2rem; }
      .stTextInput input { border-radius: 14px; padding: 14px; }
      .stButton button { border-radius: 14px; padding: 12px 18px; font-weight: 700; }
      .card {
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.04);
        border-radius: 18px;
        padding: 16px 18px;
        margin-top: 12px;
      }
      .muted { opacity: 0.75; font-size: 0.92rem; }
      .pill {
        display: inline-block;
        border-radius: 999px;
        padding: 6px 10px;
        margin-right: 6px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        font-size: 0.86rem;
      }
      code { border-radius: 10px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Header ----
left, right = st.columns([0.7, 0.3], vertical_alignment="center")
with left:
    st.title("🌿 Ayurveda AI")
    st.write("Ask questions **only from your documents** (RAG + citations).")
with right:
    st.markdown('<div class="card">⚡ Fast • Safe • Cited</div>', unsafe_allow_html=True)

# ---- Load docs from backend ----
@st.cache_data(ttl=60)
def fetch_docs():
    r = requests.get(f"{API_BASE}/list_docs", timeout=20)
    r.raise_for_status()
    return r.json()["docs"]

docs = []
docs_error = None
try:
    docs = fetch_docs()
except Exception as e:
    docs_error = str(e)

# ---- Sidebar controls ----
st.sidebar.header("⚙️ Controls")

if docs_error:
    st.sidebar.error("Backend not reachable or /docs failed.")
    st.sidebar.caption(docs_error)
    st.stop()

# Group docs by source
by_source = {}
for d in docs:
    by_source.setdefault(d["source"] or "UNKNOWN", []).append(d)

sources = sorted(by_source.keys())
source_choice = st.sidebar.selectbox("Filter by Source (optional)", ["ALL"] + sources)

filtered_docs = docs if source_choice == "ALL" else by_source.get(source_choice, [])

doc_label_to_id = {d["display_name"]: d["doc_id"] for d in filtered_docs}
doc_labels = list(doc_label_to_id.keys())

selected_doc_labels = st.sidebar.multiselect(
    "Choose documents (optional)",
    options=doc_labels,
    help="Leave empty to search ALL documents.",
)

top_k = st.sidebar.slider("Top K chunks", min_value=3, max_value=20, value=8)
strict = st.sidebar.toggle("Strict mode (no guessing)", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: For best accuracy, pick 1–3 docs for a question.")

# ---- Main query box ----
question = st.text_input("Ask your question", placeholder="e.g., What does Charaka say about healthy living?")

ask = st.button("✨ Ask", use_container_width=True)

if ask:
    if not question.strip():
        st.warning("Type a question first.")
        st.stop()

    doc_ids = [doc_label_to_id[x] for x in selected_doc_labels]

    payload = {
        "question": question.strip(),
        "top_k": int(top_k),
        "strict": bool(strict),
        "source_filter": None if source_choice == "ALL" else source_choice,
        "doc_ids": doc_ids if doc_ids else None,
    }

    with st.spinner("Thinking with citations..."):
        r = requests.post(f"{API_BASE}/ask", json=payload, timeout=120)

    if r.status_code != 200:
        st.error("API error")
        st.code(r.text)
        st.stop()

    data = r.json()

    # ---- Answer card ----
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("✅ Answer")
    st.write(data.get("answer", ""))

    st.markdown(
        f'<div class="muted">Retrieved chunks: <b>{data.get("retrieved_count", 0)}</b></div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Citations ----
    cits = data.get("citations") or []
    if cits:
        st.subheader("📌 Citations")
        for c in cits:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"**{c.get('title','')}**")
            st.markdown(
                f"<span class='pill'>{c.get('source','')}</span>"
                f"<span class='pill'>{c.get('file_name','')}</span>"
                f"<span class='pill'>Pages {c.get('page_start',0)}–{c.get('page_end',0)}</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"Section: {c.get('section','')}")
            st.caption(f"Chunk: {c.get('chunk_id','')}")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No citations returned.")
