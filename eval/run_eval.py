"""Measure retrieval and citation quality on the synthetic catalog.

Metrics:
- retrieval_hit_at_k: some top-k chunk contains the product's SKU and the correct answer.
- top1_contains_sku: the highest-ranked chunk is the one that describes the product.
- citation_page_correct: the top-ranked source is on the page that holds the product.

This does not grade LLM answer wording. It checks that the right passage is found and cited.
Run from the project root after build_sample: python -m eval.run_eval
"""
import json
import re
import statistics
import tempfile
import time
from pathlib import Path

from app.config import EMBED_MODEL, TOP_K
from app.ingest import chunk_file
from app.vector_store import VectorStore

EVAL_DIR = Path(__file__).resolve().parent
PDF_PATH = EVAL_DIR / "data" / "sample_catalog.pdf"
QUESTIONS_PATH = EVAL_DIR / "questions.json"
RESULTS_PATH = EVAL_DIR / "results.json"


def contains_value(text: str, value: str) -> bool:
    # Stops "6 months" matching "36 months" and "$19" matching "$199".
    return re.search(rf"(?<![\w.]){re.escape(value)}(?!\w)", text) is not None


def main(k: int = TOP_K) -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    chunks = chunk_file(PDF_PATH, doc_id="eval-catalog")

    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(Path(tmp), EMBED_MODEL)
        store.add(chunks)

        rows, latencies = [], []
        for q in questions:
            started = time.perf_counter()
            results = store.search(q["question"], k)
            latencies.append((time.perf_counter() - started) * 1000)

            texts = [doc.page_content for doc, _ in results]
            hit = any(q["sku"] in t and contains_value(t, q["answer"]) for t in texts)
            top_doc = results[0][0] if results else None
            top1_sku = top_doc is not None and q["sku"] in top_doc.page_content
            cited_page_ok = top_doc is not None and top_doc.metadata.get("page") == q["page"]

            rows.append(
                {
                    "id": len(rows) + 1,
                    "field": q["field"],
                    "question": q["question"],
                    "expected": q["answer"],
                    "retrieval_hit": hit,
                    "top1_contains_sku": top1_sku,
                    "citation_page_correct": cited_page_ok,
                }
            )

    total = len(rows)
    summary = {
        "questions": total,
        "chunks_indexed": len(chunks),
        "top_k": k,
        "embed_model": EMBED_MODEL,
        f"retrieval_hit_at_{k}": round(sum(r["retrieval_hit"] for r in rows) / total, 3),
        "top1_contains_sku": round(sum(r["top1_contains_sku"] for r in rows) / total, 3),
        "citation_page_correct": round(sum(r["citation_page_correct"] for r in rows) / total, 3),
        "search_latency_ms_median": round(statistics.median(latencies), 1),
        "by_field": {},
    }
    for field in sorted({r["field"] for r in rows}):
        subset = [r for r in rows if r["field"] == field]
        summary["by_field"][field] = round(sum(r["retrieval_hit"] for r in subset) / len(subset), 3)

    RESULTS_PATH.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    misses = [r for r in rows if not r["retrieval_hit"]]
    if misses:
        print(f"\n{len(misses)} misses:")
        for r in misses[:10]:
            print(f"  - {r['question']} (expected {r['expected']})")


if __name__ == "__main__":
    main()
