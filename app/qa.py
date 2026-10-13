"""Answer questions from retrieved chunks and return citations for every source used."""
from langchain_core.documents import Document

from app import config
from app.vector_store import VectorStore

NOT_FOUND = "I could not find the answer in the uploaded documents."
SNIPPET_CHARS = 300

SYSTEM_PROMPT = (
    "You answer questions using only the numbered excerpts the user provides. "
    "Cite the excerpts you rely on inline, like [1] or [2]. "
    "If the excerpts do not contain the answer, reply exactly: "
    f"\"{NOT_FOUND}\""
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


def _format_source(index: int, doc: Document, score: float) -> dict:
    return {
        "index": index,
        "source": doc.metadata.get("source"),
        "page": doc.metadata.get("page"),
        "score": round(float(score), 3),
        "snippet": doc.page_content[:SNIPPET_CHARS],
    }


def _build_prompt(question: str, hits: list[tuple[Document, float]]) -> str:
    excerpts = "\n\n".join(
        f"[{i}] (source: {doc.metadata.get('source')}, page: {doc.metadata.get('page') or 'n/a'})\n{doc.page_content}"
        for i, (doc, _) in enumerate(hits, start=1)
    )
    return f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"


def answer_question(question: str, store: VectorStore, llm=None) -> dict:
    hits = [(doc, score) for doc, score in store.search(question, config.TOP_K) if score >= config.MIN_SCORE]
    if not hits:
        return {"answer": NOT_FOUND, "mode": "none", "sources": []}

    sources = [_format_source(i, doc, score) for i, (doc, score) in enumerate(hits, start=1)]

    if llm is None:
        # No LLM configured: the best passage is the answer, with its citation attached.
        best_doc, _ = hits[0]
        return {"answer": best_doc.page_content, "mode": "extractive", "sources": sources}

    from langchain_core.messages import HumanMessage, SystemMessage

    response = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=_build_prompt(question, hits))]
    )
    return {"answer": str(response.content).strip(), "mode": "llm", "sources": sources}
