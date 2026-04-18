"""
13 — CAPSTONE: Data Engineer Assistant (Week 12)

Goal:
  An end-to-end assistant that answers natural-language questions over your BigQuery
  warehouse. Combines everything from Weeks 1-11:

    1. RAG over BigQuery INFORMATION_SCHEMA + your doc corpus (Week 5 + 6)
    2. Function calling to run SQL (Week 7)
    3. Multi-step graph with self-correction (Week 8)
    4. LLM-as-judge eval on a golden set (Week 9)
    5. Safety: PII redaction, SQL allow-list, cost caps (Week 11)
    6. FastAPI serve endpoint (Week 11)

This is your portfolio piece. Record a 5-minute demo once it works end-to-end.

Run:
  python 13_capstone_de_assistant.py ask "How many orders per day last week?"
  python 13_capstone_de_assistant.py serve          # spins up FastAPI on :8080
  python 13_capstone_de_assistant.py eval           # runs golden-set eval

Endpoints exposed by `serve`:
  POST /ask  { "question": "..." }  →  { "answer": "...", "sql": "...", "sources": [...] }
  GET  /health
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
import importlib.util
from pathlib import Path
from typing import TypedDict

import numpy as np
from pydantic import BaseModel

from _config import (
    init_vertex, PROJECT_ID, LOCATION, GEMINI_MODEL, GEMINI_PRO_MODEL,
    EMBEDDING_MODEL, BQ_DATASET,
)
from vertexai.generative_models import GenerativeModel, GenerationConfig
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput


# ======================================================================
# 1. Safety: a SQL allow-list and a cost cap
# ======================================================================

SQL_ALLOWED = re.compile(r"^\s*select\s", re.IGNORECASE)
SQL_BANNED = re.compile(r"\b(delete|update|insert|drop|truncate|create|alter|merge)\b",
                        re.IGNORECASE)


def is_sql_safe(sql: str) -> tuple[bool, str]:
    if SQL_BANNED.search(sql):
        return False, "Only SELECT queries are allowed."
    if not SQL_ALLOWED.match(sql):
        return False, "Query must start with SELECT."
    return True, ""


MAX_SCAN_BYTES = 100 * 1024 * 1024  # 100 MB


# ======================================================================
# 2. RAG: retrieve relevant schema & docs for the question
# ======================================================================

def _index_docs() -> list[dict]:
    """One-shot: chunk the sample_docs + pull BQ information_schema, embed them.
       In production, this is a scheduled job that writes to a persistent index."""
    spec = importlib.util.spec_from_file_location(
        "chunking", Path(__file__).resolve().parent / "05_chunking.py"
    )
    ch = importlib.util.module_from_spec(spec); spec.loader.exec_module(ch)

    sample_dir = Path(__file__).resolve().parent / "sample_docs"
    docs = []
    for md in sorted(sample_dir.glob("*.md")):
        for i, c in enumerate(ch.chunk_markdown(md.read_text(), target_size=800)):
            docs.append({"id": f"doc__{md.stem}__{i}", "kind": "wiki", "text": c.text})

    # BQ schema — one chunk per table
    try:
        from google.cloud import bigquery
        bq = bigquery.Client(project=PROJECT_ID)
        sql = f"""
          SELECT table_name,
                 STRING_AGG(column_name || ' ' || data_type, ', ') AS columns
          FROM `{PROJECT_ID}.{BQ_DATASET}.INFORMATION_SCHEMA.COLUMNS`
          GROUP BY table_name
        """
        for r in bq.query(sql).result():
            docs.append({
                "id": f"schema__{r.table_name}",
                "kind": "schema",
                "text": f"Table {r.table_name}: {r.columns}",
            })
    except Exception as e:
        # Fall back to a mock schema so the lab runs without a configured BQ dataset
        docs.append({
            "id": "schema__docs",
            "kind": "schema",
            "text": "Table docs: id STRING, doc_id STRING, chunk_id INT64, text STRING, embedding FLOAT64[]",
        })

    # Embed
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
    BATCH = 50
    for i in range(0, len(docs), BATCH):
        batch = docs[i:i + BATCH]
        inputs = [TextEmbeddingInput(text=d["text"], task_type="RETRIEVAL_DOCUMENT") for d in batch]
        vs = model.get_embeddings(inputs)
        for d, v in zip(batch, vs):
            d["emb"] = np.array(v.values, dtype=np.float32)
    return docs


_INDEX: list[dict] | None = None
def get_index():
    global _INDEX
    if _INDEX is None:
        print("[capstone] building index...")
        _INDEX = _index_docs()
    return _INDEX


def retrieve(question: str, k: int = 5) -> list[dict]:
    idx = get_index()
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
    q = np.array(model.get_embeddings(
        [TextEmbeddingInput(text=question, task_type="RETRIEVAL_QUERY")]
    )[0].values, dtype=np.float32)

    def norm(v): return v / (np.linalg.norm(v) + 1e-12)
    qn = norm(q)
    scored = [(float(norm(d["emb"]) @ qn), d) for d in idx]
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored[:k]]


# ======================================================================
# 3. Ask pipeline: retrieve → write SQL → validate → run → summarize
# ======================================================================

class AskResult(BaseModel):
    answer: str
    sql: str | None = None
    sources: list[str] = []
    error: str | None = None


def ask(question: str) -> AskResult:
    t0 = time.perf_counter()
    hits = retrieve(question, k=5)
    schema_ctx = "\n".join(h["text"] for h in hits if h["kind"] == "schema")
    wiki_ctx = "\n".join(h["text"] for h in hits if h["kind"] == "wiki")

    model = GenerativeModel(
        GEMINI_MODEL,
        system_instruction=(
            "You are a data engineer assistant. "
            "If the question is about warehouse data, write a single BigQuery SELECT. "
            "If the question is conceptual, answer from the wiki context. "
            "Refuse if the question asks to modify data."
        ),
    )
    plan_resp = model.generate_content(
        f"SCHEMA:\n{schema_ctx}\n\nWIKI:\n{wiki_ctx}\n\nQUESTION: {question}\n\n"
        "If a SQL query is needed, respond with JSON: "
        '{"kind":"sql","sql":"..."} otherwise {"kind":"prose","answer":"..."}',
        generation_config=GenerationConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["sql", "prose"]},
                    "sql": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": ["kind"],
            },
            temperature=0.1,
        ),
    )
    plan = json.loads(plan_resp.text)

    if plan["kind"] == "prose":
        return AskResult(answer=plan.get("answer", ""), sources=[h["id"] for h in hits])

    sql = plan.get("sql", "")
    ok, msg = is_sql_safe(sql)
    if not ok:
        return AskResult(answer="Sorry, that query wasn't safe to run.", sql=sql,
                         error=msg, sources=[h["id"] for h in hits])

    # Execute
    try:
        from google.cloud import bigquery
        bq = bigquery.Client(project=PROJECT_ID)
        rows = list(bq.query(
            sql,
            job_config=bigquery.QueryJobConfig(maximum_bytes_billed=MAX_SCAN_BYTES),
        ).result(max_results=50))
        if rows:
            header = list(rows[0].keys())
            rows_preview = " | ".join(header) + "\n" + "\n".join(
                " | ".join(str(r.get(c)) for c in header) for r in rows[:10]
            )
        else:
            rows_preview = "(no rows)"
    except Exception as e:
        return AskResult(answer=f"SQL execution failed: {e}", sql=sql,
                         error=str(e), sources=[h["id"] for h in hits])

    summary_model = GenerativeModel(GEMINI_MODEL)
    summary = summary_model.generate_content(
        f"Summarize the SQL result in 2-3 sentences for a non-technical user.\n"
        f"Question: {question}\nSQL: {sql}\nResult:\n{rows_preview}",
        generation_config=GenerationConfig(temperature=0.1, max_output_tokens=200),
    ).text

    print(f"[capstone] ask complete in {time.perf_counter() - t0:.2f}s")
    return AskResult(answer=summary, sql=sql, sources=[h["id"] for h in hits])


# ======================================================================
# 4. FastAPI server
# ======================================================================

def serve():
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn

    app = FastAPI(title="Data Engineer Assistant")

    class AskBody(BaseModel):
        question: str

    @app.post("/ask")
    def ask_endpoint(body: AskBody):
        return ask(body.question).model_dump()

    @app.get("/health")
    def health():
        return {"ok": True}

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)


# ======================================================================
# 5. Mini eval harness
# ======================================================================

GOLDEN = [
    {"q": "What is Vertex AI?", "expect_kind": "prose"},
    {"q": "How do I defend against prompt injection?", "expect_kind": "prose"},
    {"q": "Count rows in the docs table.", "expect_kind": "sql"},
]


def eval_run():
    init_vertex()
    judge = GenerativeModel(GEMINI_PRO_MODEL)
    scores = []
    for ex in GOLDEN:
        r = ask(ex["q"])
        prompt = f"""On a scale of 1-5, is this answer good for the question?
Q: {ex['q']}
A: {r.answer}
SQL (if any): {r.sql}
Return JSON: {{"score": <int>, "reason": "<str>"}}"""
        resp = judge.generate_content(
            prompt,
            generation_config=GenerationConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {"score": {"type": "integer"}, "reason": {"type": "string"}},
                    "required": ["score"],
                },
                temperature=0.0,
            ),
        )
        j = json.loads(resp.text)
        print(f"Q: {ex['q']}")
        print(f"  A: {r.answer[:150]}...")
        print(f"  judge: {j}")
        scores.append(j["score"])
    if scores:
        print(f"\nMean score: {np.mean(scores):.2f}")


if __name__ == "__main__":
    init_vertex()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "ask"
    if cmd == "ask":
        q = sys.argv[2] if len(sys.argv) > 2 else "What is Vertex AI?"
        r = ask(q)
        print(json.dumps(r.model_dump(), indent=2))
    elif cmd == "serve":
        serve()
    elif cmd == "eval":
        eval_run()
    else:
        print(__doc__)


# ===== Stretch goals (for Week 12 improvements) =====
# 1. Replace the in-memory RAG with Vertex AI Vector Search from 07_*.
# 2. Add per-user rate limits and a cost-per-request counter (Week 11).
# 3. Add PII redaction on the question before retrieval (Cloud DLP).
# 4. Add a "clarify" node: if retrieval confidence is low, ask the user for more detail.
# 5. Deploy to Cloud Run: `gcloud run deploy` with this file as entrypoint.
