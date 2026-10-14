# DocuMind

Document Q&A for PDF and DOCX files. Upload documents, ask a question, and get an answer with the passages it came from (file name and page number).

- **Backend:** FastAPI. Parses files, chunks them per page, embeds chunks, and searches a FAISS index.
- **Answers:** Claude (via LangChain) when `ANTHROPIC_API_KEY` is set. Without a key, DocuMind returns the best matching passage with its citation, so it works offline.
- **UI:** Streamlit chat that talks to the API over HTTP.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

The first run downloads the `all-MiniLM-L6-v2` embedding model (about 90 MB).

Optional, for generated answers:

```bash
set ANTHROPIC_API_KEY=your-key   # Windows (macOS/Linux: export ANTHROPIC_API_KEY=...)
```

## Run

The Streamlit app runs everything in one process:

```bash
streamlit run streamlit_app.py
```

Open the URL, upload PDFs or DOCX files in the sidebar, click **Index documents**, then ask questions.

The FastAPI routes are optional, for use from other programs:

```bash
uvicorn app.main:app --port 8000
```

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub (it can stay private).
2. At [share.streamlit.io](https://share.streamlit.io), choose **Create app**, select the repo, branch `main`, and main file `streamlit_app.py`.
3. Under **Advanced settings**, pick Python 3.10 or later. Optionally add the secret `ANTHROPIC_API_KEY = "..."` to enable generated answers and summaries.
4. Deploy. The first start downloads the embedding model, so expect a minute or two.

Notes for the hosted copy:
- Uploaded files and the index live on the app's disk. They are cleared when the app restarts or redeploys, so re-index after a restart.
- Without `ANTHROPIC_API_KEY`, fact questions still work, but summaries and question papers show the "needs a language model" message.

API endpoints:

| Method | Path         | Purpose                                              |
|--------|--------------|------------------------------------------------------|
| GET    | `/health`    | Status and whether generated answers are enabled     |
| GET    | `/documents` | Indexed files with chunk counts                      |
| POST   | `/documents` | Upload one or more files (multipart field `files`)   |
| POST   | `/ask`       | `{"question": "..."}` → answer, mode, and sources    |

## Configuration

Environment variables (all optional):

| Variable | Default | Meaning |
|---|---|---|
| `DOCUMIND_EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `DOCUMIND_CHUNK_SIZE` / `DOCUMIND_CHUNK_OVERLAP` | `500` / `50` | Characters per chunk |
| `DOCUMIND_TOP_K` | `4` | Chunks passed to the answer step |
| `DOCUMIND_MIN_SCORE` | `0.3` | Cosine similarity below which the answer is "not found" |
| `DOCUMIND_LLM_MODEL` | `claude-haiku-5-5` | Claude model used when a key is set |
| `DOCUMIND_DATA_DIR` | `./data` | Uploads, FAISS index, and manifest |

## How it works

1. **Ingest.** PDFs are read page by page with `pypdf`; DOCX paragraphs are read with `python-docx`. Each page is split separately, so every chunk keeps an exact page number.
2. **Index.** Chunks are embedded with normalised sentence-transformer vectors and stored in one FAISS index. The index and an upload manifest are saved under `data/`.
3. **Retrieve.** The question is embedded and the top-k chunks are returned with cosine similarity. Chunks below `DOCUMIND_MIN_SCORE` are dropped; if none remain, the app says it could not find the answer.
4. **Answer.** With a key, Claude gets the numbered excerpts and must cite them inline (`[1]`, `[2]`). Without a key, the top passage is returned as the answer.

## Evaluation

`eval/` contains a reproducible test on a synthetic catalogue. All products are fictional, so the expected answers are known exactly.

```bash
pip install -r requirements-dev.txt
python -m eval.build_sample      # writes eval/data/sample_catalog.pdf and eval/questions.json
python -m eval.run_eval          # writes eval/results.json
```

The set has 100 products across 20 PDF pages, with one question each: warranty, weight, price, or colour.

Current results (`all-MiniLM-L6-v2`, 500/50 chunks, top-k = 4):

| Metric | Value |
|---|---|
| Retrieval hit@4 (correct SKU and answer in top-4 chunks) | 65% |
| Top-1 chunk is the product's chunk | 35% |
| Top-1 source is on the correct page (citation accuracy) | 36% |
| Retrieval hit by field: warranty / weight / price / colour | 52% / 76% / 64% / 68% |
| Median search latency | ~30 ms |

**What this measures:** whether the right passage is retrieved and cited. It does **not** grade generated answer wording. That needs an `ANTHROPIC_API_KEY` and a separate grading step, which has not been run.

**Known weakness:** dense embeddings separate similar product names poorly (for example "Glacier Charger" and "Glacier Blender"), so many misses are a similar product on the same page. Chunk size changes the result in both directions. At 250/25 the top-4 hit rate is 69%, and at 150/0 it is 53%. The shipped default is 500/50. Those alternatives were measured on the same 100 questions, so they are not held-out results.

The `MIN_SCORE` default of 0.3 comes from the same catalogue: catalogue questions scored at least 0.35 and ten off-topic questions scored at most 0.25. A larger set of real documents would be needed to confirm it.

## Limitations

- Scanned PDFs (no text layer) are rejected. OCR is not included.
- DOCX tables and images are not extracted. DOCX has no page numbers, so citations show the file name only.
- The index is a single local FAISS file, with no user accounts and no deletion of individual documents.
- Hybrid keyword + vector search (for example BM25 with reciprocal-rank fusion) would likely fix the product-name misses above. It is not implemented.
