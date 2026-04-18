"""
04 — Embeddings basics (Week 4)

Goal:
  - Generate text embeddings with Vertex AI (text-embedding-004).
  - Store them in a numpy array.
  - Query by cosine similarity and see which docs come back.

Key intuition:
  Similar MEANINGS → similar VECTORS. That's the whole trick.

Run:
  python 04_embeddings_basics.py
"""
import numpy as np
from _config import init_vertex, EMBEDDING_MODEL
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput


DOCS = [
    "Airflow is a workflow orchestrator for data pipelines.",
    "dbt transforms data using SQL and Jinja templates.",
    "BigQuery is Google's serverless data warehouse.",
    "Snowflake is a cloud data warehouse.",
    "Kafka is a distributed event-streaming platform.",
    "Pub/Sub is Google's messaging service for event-driven systems.",
    "Terraform is infrastructure as code.",
    "Pytest is the most popular Python testing framework.",
    "An LLM predicts the next token given a prompt.",
    "Embeddings turn text into vectors where similar meanings are close.",
    "RAG augments an LLM with retrieved documents for grounding.",
    "Vertex AI offers Gemini, embeddings, and vector search.",
    "Prompt injection is when a user overrides the system prompt with hostile input.",
    "A vector database indexes embeddings for approximate nearest-neighbor search.",
    "Fine-tuning updates model weights on your labeled examples.",
]


def embed(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
    """Returns (N, 768) float array."""
    model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
    inputs = [TextEmbeddingInput(text=t, task_type=task_type) for t in texts]
    embeddings = model.get_embeddings(inputs)
    return np.array([e.values for e in embeddings], dtype=np.float32)


def normalize(vs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vs, axis=1, keepdims=True) + 1e-12
    return vs / norms


def top_k(query: str, doc_vecs: np.ndarray, docs: list[str], k: int = 3):
    q = embed([query], task_type="RETRIEVAL_QUERY")
    # Normalize → dot product == cosine similarity
    qn = normalize(q)[0]
    dn = normalize(doc_vecs)
    sims = dn @ qn
    idx = np.argsort(-sims)[:k]
    return [(docs[i], float(sims[i])) for i in idx]


if __name__ == "__main__":
    init_vertex()

    print(f"Embedding {len(DOCS)} docs...")
    doc_vecs = embed(DOCS)
    print(f"  shape: {doc_vecs.shape}  dtype: {doc_vecs.dtype}")

    queries = [
        "What tool schedules my ETL jobs?",
        "How do I search a vector index?",
        "What's the risk of user input manipulating my chatbot?",
        "Snowflake",
    ]

    for q in queries:
        print(f"\nQ: {q}")
        for doc, score in top_k(q, doc_vecs, DOCS, k=3):
            print(f"  {score:+.3f}  {doc}")


# ===== Try next =====
# 1. Try `task_type="SEMANTIC_SIMILARITY"` for everything. Note the score changes — Vertex
#    uses different prompts behind the scenes for query vs document.
# 2. Add docs in another language and query in English. text-multilingual-embedding-002
#    handles that if you switch models.
# 3. Plot the vectors with PCA or UMAP to 2D and eyeball the clustering.
