"""Answer questions from retrieved chunks and return citations for every source used."""
import re
from collections import Counter

from langchain_core.documents import Document

from app import config
from app.vector_store import VectorStore

NOT_FOUND = "I could not find the answer in the uploaded documents."
NEEDS_LLM = (
    "Question papers and quizzes need a language model, because they write new questions. "
    "Add an ANTHROPIC_API_KEY (environment variable or app secret) and restart the app. "
    "The passages it would work from are listed below."
)
SNIPPET_CHARS = 300
SUMMARY_SENTENCES = 8

# Free summaries (no LLM): pick the most representative sentences of the document.
SUMMARY_PATTERN = re.compile(r"\b(summar\w*|overview|outline|key points|main points|main topics)\b", re.IGNORECASE)
QUESTION_PATTERN = re.compile(r"\b(questions?|quiz|exam|mcqs?)\b", re.IGNORECASE)
STOPWORDS = set(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will with".split()
)

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


def is_free_summary_request(question: str) -> bool:
    return SUMMARY_PATTERN.search(question) is not None and QUESTION_PATTERN.search(question) is None


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in STOPWORDS]


def extractive_summary(chunks: list[Document], max_sentences: int) -> tuple[str, list[int]]:
    """Pick the sentences that share the most frequent words in the document.

    Returns the summary text and the positions (in `chunks`) of the chunks it came from.
    """
    # Words in more than a fifth of the chunks are running headers and footers, not content.
    chunk_counts = Counter(word for chunk in chunks for word in set(_words(chunk.page_content)))
    boilerplate = {w for w, n in chunk_counts.items() if len(chunks) >= 5 and n > 0.2 * len(chunks)}

    sentences: list[tuple[str, int]] = []
    for position, chunk in enumerate(chunks):
        for sentence in re.split(r"(?<=[.!?])\s+|\n{2,}", chunk.page_content):
            tokens = sentence.split()
            # Skip fragments, table-of-contents lines and number-heavy rows.
            if len(tokens) >= 8 and sum(t.isalpha() for t in tokens) / len(tokens) >= 0.6:
                sentences.append((sentence.strip(), position))
    if not sentences:
        return "", []

    frequency = Counter(
        word for sentence, _ in sentences for word in _words(sentence) if word not in boilerplate
    )

    def score(sentence: str) -> float:
        words = [w for w in _words(sentence) if w not in boilerplate]
        return sum(frequency[w] for w in words) / len(words) if words else 0.0

    best = sorted(range(len(sentences)), key=lambda i: score(sentences[i][0]), reverse=True)[:max_sentences]
    chosen = sorted(best)  # keep document order so the summary reads naturally
    text = "\n".join(sentences[i][0] for i in chosen)
    return text, sorted({sentences[i][1] for i in chosen})


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
        if llm is None and is_free_summary_request(question):
            text, positions = extractive_summary(chunks, SUMMARY_SENTENCES)
            sources = [_format_source(i, chunks[p], None) for i, p in enumerate(positions, start=1)]
            return {"answer": text, "mode": "summary_free", "sources": sources}
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
