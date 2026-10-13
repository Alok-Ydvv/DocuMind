"""FastAPI backend: upload documents, list them, and ask questions."""
import json
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app import config
from app.ingest import SUPPORTED_EXTENSIONS, chunk_file
from app.qa import answer_question, get_llm
from app.vector_store import VectorStore

_manifest_lock = threading.Lock()
state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load the embedding model once at startup, not per request.
    state["store"] = VectorStore(config.INDEX_DIR, config.EMBED_MODEL)
    state["llm"] = get_llm()
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    yield
    state.clear()


app = FastAPI(title="DocuMind", version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class Source(BaseModel):
    index: int
    source: str | None
    page: int | None
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    mode: str
    sources: list[Source]


def _read_manifest() -> list[dict]:
    if not config.MANIFEST_PATH.exists():
        return []
    return json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))


def _write_manifest(entries: list[dict]) -> None:
    config.MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.MANIFEST_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "llm": "claude" if state.get("llm") else "extractive"}


@app.get("/documents")
def list_documents():
    with _manifest_lock:
        return _read_manifest()


@app.post("/documents", status_code=201)
def upload_documents(files: list[UploadFile] = File(...)):
    indexed = []
    for upload in files:
        suffix = Path(upload.filename or "").suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"{upload.filename}: only PDF and DOCX are supported")

        doc_id = uuid.uuid4().hex
        saved_path = config.UPLOAD_DIR / f"{doc_id}{suffix}"
        saved_path.write_bytes(upload.file.read())

        chunks = chunk_file(saved_path, doc_id)
        if not chunks:
            saved_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=422,
                detail=f"{upload.filename}: no extractable text (scanned PDFs are not supported)",
            )
        for chunk in chunks:
            chunk.metadata["source"] = upload.filename

        state["store"].add(chunks)
        entry = {"id": doc_id, "filename": upload.filename, "chunks": len(chunks)}
        with _manifest_lock:
            manifest = _read_manifest()
            manifest.append(entry)
            _write_manifest(manifest)
        indexed.append(entry)
    return indexed


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    result = answer_question(request.question, state["store"], state.get("llm"))
    return result
