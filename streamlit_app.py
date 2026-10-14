"""Streamlit front end. Runs the document service in-process, so one process serves the whole app."""
import html

import streamlit as st

from app.service import DocumentService, UploadError

st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide", initial_sidebar_state="expanded")

# (button label, question sent to the assistant)
SUGGESTIONS = [
    ("📝 Summarize", "Summarize this document"),
    ("🧪 Question paper", "Make a question paper with 10 questions"),
    ("🔑 Key points", "What are the key points?"),
]

MODE_BADGES = {
    "llm": ("✨ Answered by Claude", "badge-ai"),
    "extractive": ("📌 Best matching passage", "badge-plain"),
    "needs_llm": ("🔒 Needs an API key", "badge-warn"),
    "none": ("🔍 Not found in documents", "badge-muted"),
}

CSS = """
<style>
.block-container { padding-top: 2rem; max-width: 1100px; }
.hero {
    padding: 1.8rem 2rem; border-radius: 18px; margin-bottom: 1.2rem;
    background: linear-gradient(135deg, #7C5CFF 0%, #2DD4BF 100%);
    color: #fff; box-shadow: 0 10px 30px rgba(124, 92, 255, 0.25);
}
.hero h1 { margin: 0; font-size: 2.3rem; font-weight: 800; letter-spacing: -0.5px; color: #fff; }
.hero p { margin: 0.4rem 0 0; font-size: 1.02rem; opacity: 0.92; }
.badge {
    display: inline-block; padding: 0.2rem 0.65rem; border-radius: 999px;
    font-size: 0.78rem; font-weight: 600; margin-right: 0.4rem; margin-bottom: 0.4rem;
}
.badge-ai { background: rgba(124, 92, 255, 0.18); color: #B9A8FF; }
.badge-plain { background: rgba(45, 212, 191, 0.16); color: #7EE8D8; }
.badge-warn { background: rgba(251, 191, 36, 0.16); color: #FCD34D; }
.badge-muted { background: rgba(148, 163, 184, 0.18); color: #CBD5E1; }
.badge-chip { background: rgba(255, 255, 255, 0.14); color: #fff; }
.source-card {
    border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 12px;
    padding: 0.85rem 1rem; margin-bottom: 0.7rem; background: rgba(255, 255, 255, 0.03);
}
.source-head { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.4rem; flex-wrap: wrap; }
.source-num {
    background: linear-gradient(135deg, #7C5CFF, #2DD4BF); color: #fff; font-weight: 700;
    border-radius: 8px; padding: 0.1rem 0.5rem; font-size: 0.85rem;
}
.source-file { font-weight: 600; }
.source-meta { opacity: 0.7; font-size: 0.85rem; }
.bar { height: 6px; border-radius: 999px; background: rgba(148, 163, 184, 0.25); margin: 0.4rem 0 0.6rem; }
.bar > span { display: block; height: 100%; border-radius: 999px; background: linear-gradient(90deg, #7C5CFF, #2DD4BF); }
.snippet { font-size: 0.88rem; opacity: 0.85; white-space: pre-wrap; }
.doc-card {
    border: 1px solid rgba(148, 163, 184, 0.25); border-radius: 12px;
    padding: 0.7rem 0.9rem; margin-bottom: 0.5rem; background: rgba(255, 255, 255, 0.03);
}
.doc-name { font-weight: 600; word-break: break-word; }
.empty {
    border: 2px dashed rgba(124, 92, 255, 0.45); border-radius: 18px; padding: 2.2rem; text-align: center;
    background: rgba(124, 92, 255, 0.05); margin-top: 1rem;
}
.empty h3 { margin-bottom: 0.4rem; }
.empty p { opacity: 0.75; margin: 0; }
.stButton > button { border-radius: 999px; }
.passage {
    white-space: pre-wrap; font-size: 0.95rem; line-height: 1.6; padding: 0.9rem 1rem;
    border-left: 3px solid #2DD4BF; border-radius: 8px; background: rgba(45, 212, 191, 0.06);
}
</style>
"""


@st.cache_resource(show_spinner="Loading the embedding model (first run only)...")
def get_service() -> DocumentService:
    return DocumentService()


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})", expanded=False):
        for src in sources:
            location = f"page {src['page']}" if src["page"] else "full text"
            bar = ""
            if src["score"] is not None:
                width = max(0, min(100, round(src["score"] * 100)))
                bar = f'<div class="bar"><span style="width:{width}%"></span></div>'
            snippet = html.escape(src["snippet"])
            st.markdown(
                f"""
<div class="source-card">
  <div class="source-head">
    <span class="source-num">{src['index']}</span>
    <span class="source-file">{html.escape(src['source'] or 'unknown')}</span>
    <span class="source-meta">· {location}</span>
  </div>
  {bar}
  <div class="snippet">{snippet}</div>
</div>
""",
                unsafe_allow_html=True,
            )


def render_mode_badge(mode: str) -> None:
    label, css_class = MODE_BADGES.get(mode, MODE_BADGES["none"])
    st.markdown(f'<span class="badge {css_class}">{label}</span>', unsafe_allow_html=True)


def render_answer(mode: str, answer: str) -> None:
    # Passages are raw document text: keep their line breaks and don't let "$" start LaTeX.
    if mode == "extractive":
        st.markdown(f'<div class="passage">{html.escape(answer)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(answer.replace("$", r"\$"))


st.markdown(CSS, unsafe_allow_html=True)
service = get_service()
docs = service.list_documents()
total_chunks = sum(d["chunks"] for d in docs)
mode_label = "Claude answers" if service.llm else "Passage mode (no API key)"

st.markdown(
    f"""
<div class="hero">
  <h1>📄 DocuMind</h1>
  <p>Ask questions about your PDF and DOCX files. Every answer points to the exact page it came from.</p>
  <div style="margin-top:0.9rem">
    <span class="badge badge-chip">📚 {len(docs)} document(s)</span>
    <span class="badge badge-chip">🧩 {total_chunks} chunks</span>
    <span class="badge badge-chip">⚡ {mode_label}</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 📁 Your documents")
    uploads = st.file_uploader(
        "Drop PDF or DOCX files here", type=["pdf", "docx"], accept_multiple_files=True
    )
    if st.button("🚀 Index documents", disabled=not uploads, use_container_width=True, type="primary"):
        with st.spinner("Reading and indexing..."):
            try:
                entries = [service.index_file(f.name, f.getvalue()) for f in uploads]
                st.toast(f"Indexed {len(entries)} file(s)", icon="✅")
                st.rerun()
            except UploadError as exc:
                st.error(str(exc))

    st.markdown(f"#### Indexed ({len(docs)})")
    if not docs:
        st.caption("Nothing indexed yet.")
    for doc in docs:
        st.markdown(
            f'<div class="doc-card"><div class="doc-name">📄 {html.escape(doc["filename"])}</div>'
            f'<div class="source-meta">{doc["chunks"]} chunks</div></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending" not in st.session_state:
    st.session_state.pending = None

if not docs:
    st.markdown(
        """
<div class="empty">
  <h3>👋 Start by uploading a document</h3>
  <p>Use the panel on the left to add a PDF or DOCX, then click <b>Index documents</b>.</p>
</div>
""",
        unsafe_allow_html=True,
    )

if docs and not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(len(SUGGESTIONS))
    for col, (label, prompt) in zip(cols, SUGGESTIONS):
        if col.button(label, use_container_width=True, key=f"suggest-{label}"):
            st.session_state.pending = prompt
            st.rerun()

for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        if message["role"] == "assistant":
            render_mode_badge(message.get("mode", "none"))
            render_answer(message.get("mode", "none"), message["content"])
        else:
            st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

question = st.chat_input("Ask anything about your documents...")
if st.session_state.pending:
    question = st.session_state.pending
    st.session_state.pending = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        if not docs:
            result = {"answer": "Upload and index a document first.", "mode": "none", "sources": []}
        else:
            with st.spinner("Searching your documents..."):
                result = service.ask(question)
        render_mode_badge(result["mode"])
        render_answer(result["mode"], result["answer"])
        render_sources(result["sources"])

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "mode": result["mode"], "sources": result["sources"]}
    )
