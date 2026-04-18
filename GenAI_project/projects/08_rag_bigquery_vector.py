"""
08 — RAG with BigQuery VECTOR_SEARCH (Weeks 5–6)

Goal:
  - When your data already lives in BigQuery, keep the vector index there too.
  - Use BQ ML to generate embeddings server-side OR pass them in from Vertex AI.
  - Query with VECTOR_SEARCH() SQL function.

Why this matters:
  As a data engineer, your users' source of truth is often BigQuery. Keeping the index
  co-located with the data avoids a second system to ETL into. It also means governance
  (IAM, dataset policies, VPC-SC) is inherited automatically.

Run:
  python 08_rag_bigquery_vector.py setup    # create table, load sample data, build index
  python 08_rag_bigquery_vector.py query "your question"
"""
from __future__ import annotations
import sys
import importlib.util
from pathlib import Path

from google.cloud import bigquery
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.generative_models import GenerativeModel, GenerationConfig

from _config import init_vertex, PROJECT_ID, LOCATION, GEMINI_MODEL, EMBEDDING_MODEL, BQ_DATASET


TABLE = f"{PROJECT_ID}.{BQ_DATASET}.docs"


def _chunks_for_bq():
    spec = importlib.util.spec_from_file_location(
        "chunking", Path(__file__).resolve().parent / "05_chunking.py"
    )
    ch = importlib.util.module_from_spec(spec); spec.loader.exec_module(ch)
    sample_dir = Path(__file__).resolve().parent / "sample_docs"
    rows = []
    for md in sorted(sample_dir.glob("*.md")):
        for i, c in enumerate(ch.chunk_markdown(md.read_text(), target_size=800)):
            rows.append({"id": f"{md.stem}__{i}", "doc_id": md.stem,
                         "chunk_id": i, "text": c.text})
    return rows


def setup():
    bq = bigquery.Client(project=PROJECT_ID, location=LOCATION)

    # 1. Dataset
    dataset_ref = bigquery.Dataset(f"{PROJECT_ID}.{BQ_DATASET}")
    dataset_ref.location = LOCATION
    bq.create_dataset(dataset_ref, exists_ok=True)
    print(f"Dataset ok: {BQ_DATASET}")

    # 2. Table
    schema = [
        bigquery.SchemaField("id", "STRING"),
        bigquery.SchemaField("doc_id", "STRING"),
        bigquery.SchemaField("chunk_id", "INT64"),
        bigquery.SchemaField("text", "STRING"),
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
    ]
    bq.create_table(bigquery.Table(TABLE, schema=schema), exists_ok=True)
    print(f"Table ok: {TABLE}")

    # 3. Embed locally (with Vertex AI) and load into BQ
    rows = _chunks_for_bq()
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
    BATCH = 100
    for i in range(0, len(rows), BATCH):
        batch = rows[i:i + BATCH]
        inputs = [TextEmbeddingInput(text=r["text"], task_type="RETRIEVAL_DOCUMENT") for r in batch]
        embs = model.get_embeddings(inputs)
        for r, e in zip(batch, embs):
            r["embedding"] = e.values
    # Replace existing rows for idempotency
    bq.query(f"TRUNCATE TABLE `{TABLE}`").result()
    errors = bq.insert_rows_json(TABLE, rows)
    if errors:
        raise RuntimeError(errors)
    print(f"Inserted {len(rows)} rows")

    # 4. Create a vector index (requires >= 5,000 rows in real use; BQ will warn for small sets)
    try:
        bq.query(f"""
          CREATE OR REPLACE VECTOR INDEX docs_idx
          ON `{TABLE}`(embedding)
          OPTIONS(distance_type='COSINE', index_type='IVF')
        """).result()
        print("Vector index created.")
    except Exception as e:
        print(f"(Index creation skipped or failed: {e})")
        print("  For small datasets, BQ can still run VECTOR_SEARCH without an index.")


def query(question: str, top_k: int = 4):
    bq = bigquery.Client(project=PROJECT_ID, location=LOCATION)
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
    qvec = model.get_embeddings(
        [TextEmbeddingInput(text=question, task_type="RETRIEVAL_QUERY")]
    )[0].values

    sql = f"""
    SELECT
      base.id, base.doc_id, base.text,
      distance
    FROM VECTOR_SEARCH(
      TABLE `{TABLE}`, 'embedding',
      (SELECT @q AS embedding),
      top_k => {top_k},
      distance_type => 'COSINE'
    )
    """
    job = bq.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("q", "FLOAT64", qvec)]
    ))
    hits = list(job.result())
    ctx = "\n\n".join(f"[{h.id}]\n{h.text}" for h in hits)

    gen = GenerativeModel(
        GEMINI_MODEL,
        system_instruction=(
            "Answer using only the SOURCES. Cite as [id]. "
            "If insufficient, say you don't know."
        ),
    )
    resp = gen.generate_content(
        f"SOURCES:\n{ctx}\n\nQUESTION: {question}",
        generation_config=GenerationConfig(temperature=0.1, max_output_tokens=500),
    )
    print("\n--- hits ---")
    for h in hits:
        print(f"  {h.distance:.3f}  {h.id}")
    print("\n--- answer ---\n" + resp.text)


if __name__ == "__main__":
    init_vertex()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "query"
    if cmd == "setup":
        setup()
    else:
        q = sys.argv[2] if len(sys.argv) > 2 else "How do I defend against prompt injection?"
        query(q)


# ===== Try next =====
# 1. Add a metadata filter (e.g. WHERE doc_id = 'vertex_ai_overview') to combine SQL
#    predicates with vector search.
# 2. Use BQ's ML.GENERATE_EMBEDDING directly in SQL (skip the Python round-trip entirely).
# 3. Add a timestamp column and a "freshness decay" factor to the distance.
