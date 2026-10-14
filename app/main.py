"""FastAPI backend: upload documents, list them, and ask questions. Optional: the Streamlit UI runs without it."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.service import DocumentService, UploadError

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["service"] = DocumentService()
    yield
    state.clear()


app = FastAPI(title="DocuMind", version="1.0.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class Source(BaseModel):
    index: int
    source: str | None
    page: int | None
    score: float | None
    snippet: str


class AskResponse(BaseModel):
    answer: str
    mode: str
    sources: list[Source]


@app.get("/health")
def health():
    return {"status": "ok", "llm": state["service"].llm_mode}


@app.get("/documents")
def list_documents():
    return state["service"].list_documents()


@app.post("/documents", status_code=201)
def upload_documents(files: list[UploadFile] = File(...)):
    try:
        return [state["service"].index_file(f.filename, f.file.read()) for f in files]
    except UploadError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc))


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    return state["service"].ask(request.question)
