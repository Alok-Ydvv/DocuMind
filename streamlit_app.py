"""Streamlit front end. Runs the document service in-process, so one process serves the whole app."""
import streamlit as st

from app.service import DocumentService, UploadError

st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading the embedding model (first run only)...")
def get_service() -> DocumentService:
    return DocumentService()


def show_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for src in sources:
            location = f"page {src['page']}" if src["page"] else "document"
            similarity = "" if src["score"] is None else f" · similarity {src['score']}"
            st.markdown(f"**[{src['index']}] {src['source']}, {location}**{similarity}")
            st.caption(src["snippet"])


service = get_service()

st.title("DocuMind")
st.caption("Ask questions about your PDF and DOCX files. Every answer lists the passages it came from.")

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"], accept_multiple_files=True)
    if st.button("Index documents", disabled=not uploads, use_container_width=True):
        with st.spinner("Reading and indexing..."):
            try:
                entries = [service.index_file(f.name, f.getvalue()) for f in uploads]
                st.success(f"Indexed {len(entries)} file(s)")
            except UploadError as exc:
                st.error(str(exc))

    docs = service.list_documents()
    st.subheader(f"Indexed ({len(docs)})")
    for doc in docs:
        st.write(f"📄 {doc['filename']} · {doc['chunks']} chunks")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            show_sources(message.get("sources", []))

question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not docs:
            answer, sources = "Upload and index a document first.", []
        else:
            with st.spinner("Searching..."):
                result = service.ask(question)
            answer, sources = result["answer"], result["sources"]
        st.markdown(answer)
        show_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
