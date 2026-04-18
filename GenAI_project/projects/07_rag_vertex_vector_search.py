"""
07 — RAG with Vertex AI Vector Search (Weeks 5–6)

Goal:
  - Move your in-memory index to Vertex AI Vector Search (managed ANN).
  - Understand: index, index endpoint, deployment.
  - Know the knobs: approximate_neighbors_count, distance_measure, namespaces/filters.

COST WARNING:
  A deployed index endpoint is billed hourly. DELETE it when you're done for the day.

This script provides two paths:
  (A) Create index + endpoint + deploy (one-time, ~30–60 min setup)
  (B) Query an already-deployed index (fast, cheap)

Run:
  python 07_rag_vertex_vector_search.py create    # (A) — rare
  python 07_rag_vertex_vector_search.py query "your question"  # (B) — most of the time
  python 07_rag_vertex_vector_search.py cleanup   # delete endpoint to stop billing
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from typing import Iterable

from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.generative_models import GenerativeModel, GenerationConfig

from _config import (
    init_vertex, PROJECT_ID, LOCATION, GEMINI_MODEL, EMBEDDING_MODEL,
    STAGING_BUCKET, VECTOR_INDEX_ID, VECTOR_INDEX_ENDPOINT_ID,
)

INDEX_DISPLAY_NAME = "ai-eng-lab-index"
ENDPOINT_DISPLAY_NAME = "ai-eng-lab-endpoint"
DEPLOYED_INDEX_ID = "ai_eng_lab_deployed"


# ------------------- ingest utilities -------------------

def _load_chunks():
    from pathlib import Path
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "chunking", Path(__file__).resolve().parent / "05_chunking.py"
    )
    ch = importlib.util.module_from_spec(spec); spec.loader.exec_module(ch)
    sample_dir = Path(__file__).resolve().parent / "sample_docs"
    items = []
    for md in sorted(sample_dir.glob("*.md")):
        for i, c in enumerate(ch.chunk_markdown(md.read_text(), target_size=800)):
            items.append({"id": f"{md.stem}__{i}", "text": c.text, "doc_id": md.stem})
    return items


def _embed_all(texts: list[str]) -> list[list[float]]:
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
    out = []
    BATCH = 100
    for i in range(0, len(texts), BATCH):
        batch = [TextEmbeddingInput(text=t, task_type="RETRIEVAL_DOCUMENT")
                 for t in texts[i:i + BATCH]]
        out.extend([e.values for e in model.get_embeddings(batch)])
    return out


# ------------------- (A) create + deploy -------------------

def create_and_deploy():
    assert STAGING_BUCKET.startswith("gs://"), "Set STAGING_BUCKET in .env (gs://...)"
    aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=STAGING_BUCKET)

    items = _load_chunks()
    print(f"Embedding {len(items)} chunks...")
    vecs = _embed_all([it["text"] for it in items])
    for it, v in zip(items, vecs):
        it["embedding"] = v

    # Write JSONL to GCS (Vertex AI expects this format for batch updates)
    import tempfile, os
    from google.cloud import storage
    local = Path(tempfile.mkdtemp()) / "embeddings.jsonl"
    with local.open("w") as f:
        for it in items:
            f.write(json.dumps({
                "id": it["id"],
                "embedding": it["embedding"],
                "restricts": [{"namespace": "doc_id", "allow_list": [it["doc_id"]]}],
            }) + "\n")
    # Upload to GCS
    client = storage.Client(project=PROJECT_ID)
    bucket_name = STAGING_BUCKET.removeprefix("gs://").split("/")[0]
    bucket = client.bucket(bucket_name)
    blob = bucket.blob("ai_eng_labs/embeddings.jsonl")
    blob.upload_from_filename(local)
    contents_uri = f"gs://{bucket_name}/ai_eng_labs/"
    print(f"Uploaded to {contents_uri}")

    print("Creating index (this takes ~20 min)...")
    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=INDEX_DISPLAY_NAME,
        contents_delta_uri=contents_uri,
        dimensions=768,
        approximate_neighbors_count=20,
        distance_measure_type="DOT_PRODUCT_DISTANCE",  # use with normalized vectors
        leaf_node_embedding_count=500,
        leaf_nodes_to_search_percent=7,
        description="AI Engineer labs",
    )
    print(f"Index resource: {index.resource_name}")

    print("Creating index endpoint...")
    endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name=ENDPOINT_DISPLAY_NAME,
        public_endpoint_enabled=True,
    )
    print(f"Endpoint resource: {endpoint.resource_name}")

    print("Deploying (this takes ~30 min)...")
    endpoint.deploy_index(index=index, deployed_index_id=DEPLOYED_INDEX_ID)
    print("Done. Copy these IDs into your .env:")
    print(f"  VECTOR_INDEX_ID={index.resource_name.split('/')[-1]}")
    print(f"  VECTOR_INDEX_ENDPOINT_ID={endpoint.resource_name.split('/')[-1]}")


# ------------------- (B) query -------------------

SYSTEM = """You are a precise technical assistant.
Answer the user's question using ONLY the sources below.
Cite sources inline as [id]. If sources are insufficient, say you don't know."""


def query(question: str, top_k: int = 4):
    assert VECTOR_INDEX_ENDPOINT_ID, "Set VECTOR_INDEX_ENDPOINT_ID in .env (after `create`)"
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    endpoint = aiplatform.MatchingEngineIndexEndpoint(VECTOR_INDEX_ENDPOINT_ID)

    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
    qvec = model.get_embeddings(
        [TextEmbeddingInput(text=question, task_type="RETRIEVAL_QUERY")]
    )[0].values

    results = endpoint.find_neighbors(
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[qvec],
        num_neighbors=top_k,
    )
    hits = results[0]  # list of MatchNeighbor

    # Map chunk IDs back to text — in production you store this in a DB/BQ.
    # For this lab we just re-chunk on the fly (slow, but self-contained).
    items = {it["id"]: it for it in _load_chunks()}
    ctx = "\n\n".join(f"[{h.id}]\n{items.get(h.id, {}).get('text', '(missing)')}" for h in hits)

    gen = GenerativeModel(GEMINI_MODEL, system_instruction=SYSTEM)
    resp = gen.generate_content(
        f"SOURCES:\n{ctx}\n\nQUESTION: {question}",
        generation_config=GenerationConfig(temperature=0.1, max_output_tokens=500),
    )
    print("\n--- hits ---")
    for h in hits:
        print(f"  {h.distance:+.3f}  {h.id}")
    print("\n--- answer ---\n" + resp.text)


# ------------------- cleanup -------------------

def cleanup():
    """Un-deploy and delete the endpoint so you stop getting billed hourly."""
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    if VECTOR_INDEX_ENDPOINT_ID:
        ep = aiplatform.MatchingEngineIndexEndpoint(VECTOR_INDEX_ENDPOINT_ID)
        for dep in ep.deployed_indexes:
            print(f"Undeploying {dep.id}...")
            ep.undeploy_index(deployed_index_id=dep.id)
        ep.delete()
        print("Endpoint deleted.")
    if VECTOR_INDEX_ID:
        aiplatform.MatchingEngineIndex(VECTOR_INDEX_ID).delete()
        print("Index deleted.")


if __name__ == "__main__":
    init_vertex()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "query"
    if cmd == "create":
        create_and_deploy()
    elif cmd == "cleanup":
        cleanup()
    else:
        q = sys.argv[2] if len(sys.argv) > 2 else "What is Vertex AI?"
        query(q)


# ===== Try next =====
# 1. Add a `restricts` namespace filter (by doc_id) at query time — only retrieve chunks
#    that match a specific document.
# 2. Switch `distance_measure_type` to COSINE_DISTANCE and compare results.
# 3. Benchmark query latency p50/p95 with `time.perf_counter()` over 100 calls.
