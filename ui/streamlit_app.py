"""Streamlit front end. Talks to the FastAPI backend over HTTP."""
import os

import requests
import streamlit as st

API_URL = os.getenv("DOCUMIND_API_URL", "http://localhost:8000")

st.set_page_config(page_title="DocuMind", page_icon="📄", layout="wide")


def api_get(path: str):
    return requests.get(f"{API_URL}{path}", timeout=30)


def show_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for src in sources:
            location = f"page {src['page']}" if src["page"] else "document"
            st.markdown(f"**[{src['index']}] {src['source']}, {location}** · similarity {src['score']}")
            st.caption(src["snippet"])


st.title("DocuMind")
st.caption("Ask questions about your PDF and DOCX files. Every answer lists the passages it came from.")

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"], accept_multiple_files=True)
    if st.button("Index documents", disabled=not uploads, use_container_width=True):
        with st.spinner("Reading and indexing..."):
            files = [("files", (f.name, f.getvalue())) for f in uploads]
            try:
                resp = requests.post(f"{API_URL}/documents", files=files, timeout=600)
                if resp.ok:
                    st.success(f"Indexed {len(resp.json())} file(s)")
                else:
                    st.error(resp.json().get("detail", resp.text))
            except requests.ConnectionError:
                st.error(f"Cannot reach the API at {API_URL}. Is uvicorn running?")

    try:
        docs = api_get("/documents").json()
        st.subheader(f"Indexed ({len(docs)})")
        for doc in docs:
            st.write(f"📄 {doc['filename']} · {doc['chunks']} chunks")
    except requests.ConnectionError:
        st.warning("API not reachable")
        docs = []

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
                try:
                    resp = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=120)
                    resp.raise_for_status()
                    body = resp.json()
                    answer, sources = body["answer"], body["sources"]
                except requests.RequestException as exc:
                    answer, sources = f"Request failed: {exc}", []
        st.markdown(answer)
        show_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
