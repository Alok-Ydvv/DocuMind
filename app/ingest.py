"""Read PDF/DOCX files and split them into chunks that keep their page number."""
from pathlib import Path

from docx import Document as DocxFile
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import CHUNK_OVERLAP, CHUNK_SIZE

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}

_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def load_pages(path: Path) -> list[tuple[int | None, str]]:
    """Return (page_number, text) pairs. PDFs give one entry per page; DOCX has no pages, so it gives one entry."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, start=1)]
    if suffix == ".docx":
        document = DocxFile(str(path))
        text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
        return [(None, text)]
    raise ValueError(f"Unsupported file type: {suffix}")


def chunk_file(path: Path, doc_id: str) -> list[Document]:
    """Split one file into chunks. Splitting happens per page so every chunk maps to exactly one page."""
    chunks: list[Document] = []
    for page, text in load_pages(path):
        if not text.strip():
            continue
        for piece in _splitter.split_text(text):
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={"doc_id": doc_id, "source": path.name, "page": page},
                )
            )
    return chunks
