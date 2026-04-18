"""
06 — In-memory RAG (Week 5)

Goal:
  - End-to-end RAG with numpy as the vector store.
  - Build the full loop: ingest → chunk → embed → retrieve → grounded prompt → answer.
  - See the difference between "answer from context" vs "answer from model's memory".

Run:
  python 06_rag_inmemory.py
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import importlib.util
import numpy as np

from _config import init_vertex, GEMINI_MODEL, EMBEDDING_MODEL
from vertexai.generative_models import GenerativeModel, GenerationConfig
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

# Module names starting with a digit can't be imported normally — load by path.
_spec = importlib.util.spec_from_file_location(
    "chunking", Path(__file__).resolve().parent / "05_chunking.py"
)
chunking = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chunking)
chunk_markdown = chunking.chunk_markdown


@dataclass
class IndexedChunk:
    doc_id: str
    chunk_id: int
    text: str
    vec: np.ndarray  # (768,)


def ingest(dir_path: Path) -> list[IndexedChunk]:
    print(f"Ingesting from {dir_path}...")
    embed_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
    items: list[IndexedChunk] = []
    for md_file in sorted(dir_path.glob("*.md")):
        text = md_file.read_text()
        chunks = chunk_markdown(text, target_size=800)
        texts = [c.text for c in chunks]
        # Batch embed
        inputs = [TextEmbeddingInput(text=t, task_type="RETRIEVAL_DOCUMENT") for t in texts]
        vecs = embed_model.get_embeddings(inputs)
        for i, (c, v) in enumerate(zip(chunks, vecs)):
            items.append(IndexedChunk(md_file.stem, i, c.text, np.array(v.values, dtype=np.float32)))
    print(f"  indexed {len(items)} chunks from {len(list(dir_path.glob('*.md')))} docs")
    return items


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def retrieve(query: str, index: list[IndexedChunk], k: int = 4) -> list[tuple[float, IndexedChunk]]:
    embed_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)
    q_vec = embed_model.get_embeddings(
        [TextEmbeddingInput(text=query, task_type="RETRIEVAL_QUERY")]
    )[0].values
    q = _normalize(np.array(q_vec, dtype=np.float32))
    scored = [(float(_normalize(it.vec) @ q), it) for it in index]
    scored.sort(key=lambda x: -x[0])
    return scored[:k]


SYSTEM = """You are a precise technical assistant.
Answer the user's question using ONLY the sources below.
- Cite sources inline as [doc_id#chunk_id].
- If the sources do not answer the question, say: "I don't know based on the provided sources."
Do not speculate."""


def answer(query: str, index: list[IndexedChunk]) -> str:
    hits = retrieve(query, index, k=4)
    ctx = "\n\n".join(
        f"[{h.doc_id}#{h.chunk_id}] (score={s:.3f})\n{h.text}" for s, h in hits
    )
    model = GenerativeModel(GEMINI_MODEL, system_instruction=SYSTEM)
    resp = model.generate_content(
        f"SOURCES:\n{ctx}\n\nQUESTION: {query}",
        generation_config=GenerationConfig(temperature=0.1, max_output_tokens=600),
    )
    return resp.text


if __name__ == "__main__":
    init_vertex()
    sample_dir = Path(__file__).resolve().parent / "sample_docs"
    index = ingest(sample_dir)

    questions = [
        "What's the difference between Gemini Flash and Pro?",
        "How do I defend against prompt injection?",
        "When should I NOT use RAG?",
        "How do I set up auth for BigQuery in Mumbai?",  # out of scope → should refuse
    ]
    for q in questions:
        print("\n" + "=" * 60)
        print(f"Q: {q}")
        print(answer(q, index))


# ===== Try next =====
# 1. Drop in a known "I don't know" question. Verify Gemini refuses and doesn't hallucinate.
# 2. Swap in the chunk_fixed strategy from 05_chunking.py. Watch quality drop on boundaries.
# 3. Add a metadata filter: only retrieve chunks whose doc_id matches a tag in the query.
# 4. Persist the index to disk with numpy.savez so subsequent runs don't re-embed.
