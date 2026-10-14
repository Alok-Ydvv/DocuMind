"""Shared document logic used by both the FastAPI routes and the Streamlit UI."""
import json
import threading
import uuid
from pathlib import Path

from app import config
from app.ingest import SUPPORTED_EXTENSIONS, chunk_file
from app.qa import answer_question, get_llm
from app.vector_store import VectorStore


class UploadError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class DocumentService:
    def __init__(self):
        # Loading the embedding model is the slow part; create one service per process.
        self.store = VectorStore(config.INDEX_DIR, config.EMBED_MODEL)
        self.llm = get_llm()
        self._manifest_lock = threading.Lock()
        config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def llm_mode(self) -> str:
        return "claude" if self.llm else "extractive"

    def _read_manifest(self) -> list[dict]:
        if not config.MANIFEST_PATH.exists():
            return []
        return json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))

    def list_documents(self) -> list[dict]:
        with self._manifest_lock:
            return self._read_manifest()

    def index_file(self, filename: str, content: bytes) -> dict:
        suffix = Path(filename or "").suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise UploadError(f"{filename}: only PDF and DOCX are supported", 400)

        doc_id = uuid.uuid4().hex
        saved_path = config.UPLOAD_DIR / f"{doc_id}{suffix}"
        saved_path.write_bytes(content)

        chunks = chunk_file(saved_path, doc_id)
        if not chunks:
            saved_path.unlink(missing_ok=True)
            raise UploadError(f"{filename}: no extractable text (scanned PDFs are not supported)", 422)
        for chunk in chunks:
            chunk.metadata["source"] = filename

        self.store.add(chunks)
        entry = {"id": doc_id, "filename": filename, "chunks": len(chunks)}
        with self._manifest_lock:
            manifest = self._read_manifest()
            manifest.append(entry)
            config.MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
            config.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return entry

    def ask(self, question: str) -> dict:
        return answer_question(question, self.store, self.llm)
