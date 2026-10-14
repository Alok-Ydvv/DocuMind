"""Answer questions from retrieved chunks and return citations for every source used."""
import re

from langchain_core.documents import Document

from app import config
from app.vector_store import VectorStore

NOT_FOUND = "I could not find the answer in the uploaded documents."
NEEDS_LLM = (
    "This request (a summary, question paper or quiz) needs a language model. "
    "Add an ANTHROPIC_API_KEY (environment variable or app secret) and restart the app. "
    "The passages it would work from are listed below."
)
SNIPPET_CHARS = 300

# Requests about the whole document. Similarity search can't match these, so they skip the threshold.
TASK_PATTERN = re.compile(
    r"\b(summar\w*|overview|outline|question paper|question bank|quiz|exam|mcqs?|key points|main points|main topics"
    r"|(make|generate|create|prepare|write)\b.*\bquestions?)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = (
    "You answer questions using only the numbered excerpts the user provides. "
    "Cite the excerpts you rely on inline, like [1] or [2]. "
    "If the excerpts do not contain the answer, reply exactly: "
    f"\"{NOT_FOUND}\""
)

TASK_SYSTEM_PROMPT = (
    "You carry out requests about a document using only the numbered excerpts the user provides. "
    "The excerpts are sampled across the whole document, so say so if the request needs more coverage. "
    "Cite excerpts inline, like [1], where they support a point."
)


def get_llm():
    """Return a Claude chat model if an API key is configured, otherwise None (extractive mode)."""
    if not config.ANTHROPIC_API_KEY:
        return None
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=config.LLM_MODEL,
        api_key=config.ANTHROPIC_API_KEY,
        temperature=0,
        max_tokens=600,
    )


def _format_source(index: int, doc: Document, score: float | None) -> dict:
    return {
        "index": index,
        "source": doc.metadata.get("source"),
        "page": doc.metadata.get("page"),
        "score": None if score is None else round(float(score), 3),
        "snippet": doc.page_content[:SNIPPET_CHARS],
    }


def is_document_task(question: str) -> bool:
    return TASK_PATTERN.search(question) is not None


def _spread_sample(chunks: list[Document], count: int) -> list[Document]:
    # Evenly spaced chunks from start to end, so a summary sees the whole document, not only its first pages.
    if len(chunks) <= count:
        return chunks
    step = (len(chunks) - 1) / (count - 1)
    return [chunks[round(i * step)] for i in range(count)]


def _build_prompt(question: str, hits: list[tuple[Document, float | None]]) -> str:
    excerpts = "\n\n".join(
        f"[{i}] (source: {doc.metadata.get('source')}, page: {doc.metadata.get('page') or 'n/a'})\n{doc.page_content}"
        for i, (doc, _) in enumerate(hits, start=1)
    )
    return f"Excerpts:\n\n{excerpts}\n\nRequest: {question}"


def _answer_with_llm(llm, system_prompt: str, question: str, hits) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    response = llm.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=_build_prompt(question, hits))]
    )
    return str(response.content).strip()


def answer_question(question: str, store: VectorStore, llm=None) -> dict:
    if is_document_task(question):
        chunks = store.all_chunks()
        if not chunks:
            return {"answer": NOT_FOUND, "mode": "none", "sources": []}
        hits = [(doc, None) for doc in _spread_sample(chunks, config.TASK_CONTEXT_CHUNKS)]
        sources = [_format_source(i, doc, None) for i, (doc, _) in enumerate(hits, start=1)]
        if llm is None:
            return {"answer": NEEDS_LLM, "mode": "needs_llm", "sources": sources}
        answer = _answer_with_llm(llm, TASK_SYSTEM_PROMPT, question, hits)
        return {"answer": answer, "mode": "llm", "sources": sources}

    hits = [(doc, score) for doc, score in store.search(question, config.TOP_K) if score >= config.MIN_SCORE]
    if not hits:
        return {"answer": NOT_FOUND, "mode": "none", "sources": []}

    sources = [_format_source(i, doc, score) for i, (doc, score) in enumerate(hits, start=1)]

    if llm is None:
        # No LLM configured: the best passage is the answer, with its citation attached.
        best_doc, _ = hits[0]
        return {"answer": best_doc.page_content, "mode": "extractive", "sources": sources}

    answer = _answer_with_llm(llm, SYSTEM_PROMPT, question, hits)
    return {"answer": answer, "mode": "llm", "sources": sources}
