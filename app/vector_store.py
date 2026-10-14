"""FAISS index wrapper. One index holds chunks from every uploaded document."""
import threading
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


class VectorStore:
    def __init__(self, index_dir: Path, model_name: str):
        self._index_dir = index_dir
        self._index_dir.mkdir(parents=True, exist_ok=True)
        # Unit-length embeddings: L2 distance and cosine similarity give the same ranking.
        self._embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
        self._lock = threading.Lock()
        self._db: FAISS | None = None
        if (index_dir / "index.faiss").exists():
            self._db = FAISS.load_local(
                str(index_dir),
                self._embeddings,
                allow_dangerous_deserialization=True,  # index is written by this app, not user-supplied
            )

    def add(self, chunks: list[Document]) -> list[str]:
        """Index chunks and return their ids, so a document can be removed later."""
        if not chunks:
            return []
        with self._lock:
            if self._db is None:
                self._db = FAISS.from_documents(chunks, self._embeddings)
                ids = list(self._db.index_to_docstore_id.values())
            else:
                ids = self._db.add_documents(chunks)
            self._db.save_local(str(self._index_dir))
        return ids

    def delete(self, ids: list[str]) -> None:
        if self._db is None or not ids:
            return
        with self._lock:
            self._db.delete(ids)
            self._db.save_local(str(self._index_dir))

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Fallback for documents indexed before chunk ids were recorded."""
        if self._db is None:
            return
        ids = [cid for cid, doc in self._db.docstore._dict.items() if doc.metadata.get("doc_id") == doc_id]
        self.delete(ids)

    def all_chunks(self) -> list[Document]:
        """Every indexed chunk in upload order. Used for whole-document tasks such as summaries."""
        if self._db is None:
            return []
        return list(self._db.docstore._dict.values())

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Return (chunk, cosine similarity in [-1, 1]), most similar first."""
        if self._db is None or self._db.index.ntotal == 0:
            return []
        # FAISS returns squared L2 distance d. For unit vectors d = 2 - 2*cos, so cos = 1 - d/2.
        return [
            (doc, 1.0 - squared_distance / 2.0)
            for doc, squared_distance in self._db.similarity_search_with_score(query, k=k)
        ]
