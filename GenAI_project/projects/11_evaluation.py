"""
11 — Evaluation harness: LLM-as-judge + automatic metrics (Week 9)

Goal:
  - Build a small golden set of Q&A for the RAG system from Week 5.
  - Score outputs with (a) automatic metrics (exact-match, ROUGE, embedding similarity)
    and (b) LLM-as-judge against your own rubric.
  - Log per-example results so you can diff across runs.

Key principle:
  Evals are tests. Commit your golden set to git. Track scores over time like you
  track test pass-rates.

Run:
  python 11_evaluation.py
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np

from _config import init_vertex, GEMINI_MODEL, GEMINI_PRO_MODEL, EMBEDDING_MODEL
from vertexai.generative_models import GenerativeModel, GenerationConfig
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput


# ---------- Golden set ----------
# In real life this lives in a CSV or BigQuery table with versioning.

GOLDEN = [
    {
        "id": "cost_1",
        "question": "When should I choose Gemini Flash over Pro?",
        "reference": (
            "Use Flash for simple, high-volume tasks (classification, extraction, "
            "simple RAG). Use Pro for complex reasoning. Flash is 5-10x cheaper."
        ),
    },
    {
        "id": "injection_1",
        "question": "What is prompt injection and how do I defend against it?",
        "reference": (
            "Prompt injection is when untrusted input overrides the system instructions. "
            "Defend by separating trust zones, sanitizing inputs, validating outputs, "
            "and using Model Armor."
        ),
    },
    {
        "id": "rag_limits_1",
        "question": "When should you NOT use RAG?",
        "reference": (
            "When the answer is deterministic in a structured source (query the DB), "
            "when the task is creative writing, or when domain vocabulary is so unusual "
            "that prompting fails — fine-tune instead."
        ),
    },
    {
        "id": "outofscope_1",
        "question": "What's the airspeed velocity of an unladen swallow?",
        "reference": "I don't know based on the provided sources.",
    },
]


# ---------- The system under test (copy of the Week 5 RAG, inlined for eval isolation) ----------

def system_under_test(question: str, index_text: str) -> str:
    model = GenerativeModel(
        GEMINI_MODEL,
        system_instruction=(
            "Answer using only the context. Cite doc ids in [brackets]. "
            "If insufficient, say: 'I don't know based on the provided sources.'"
        ),
    )
    resp = model.generate_content(
        f"SOURCES:\n{index_text}\n\nQUESTION: {question}",
        generation_config=GenerationConfig(temperature=0.1, max_output_tokens=400),
    )
    return resp.text


def _load_corpus() -> str:
    """Concatenate all sample docs. Cheap eval approach — in real life you'd run a real RAG."""
    sample_dir = Path(__file__).resolve().parent / "sample_docs"
    return "\n\n".join(f"[{p.stem}]\n{p.read_text()}" for p in sorted(sample_dir.glob("*.md")))


# ---------- Metrics ----------

def exact_match(pred: str, ref: str) -> float:
    return 1.0 if pred.strip().lower() == ref.strip().lower() else 0.0


def rouge_l(pred: str, ref: str) -> float:
    try:
        from rouge_score import rouge_scorer
        s = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return s.score(ref, pred)["rougeL"].fmeasure
    except ImportError:
        return float("nan")


def emb_sim(pred: str, ref: str, embed_model) -> float:
    vs = embed_model.get_embeddings([
        TextEmbeddingInput(text=pred, task_type="SEMANTIC_SIMILARITY"),
        TextEmbeddingInput(text=ref, task_type="SEMANTIC_SIMILARITY"),
    ])
    a = np.array(vs[0].values); b = np.array(vs[1].values)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


JUDGE_PROMPT = """You are grading an AI assistant's answer.

Score from 1 to 5 on each criterion. Be strict. Return ONLY JSON.

CRITERIA:
- factuality: does the answer match the reference?
- completeness: does it cover the key points?
- grounding: does it cite sources (or appropriately say 'I don't know')?

QUESTION:
{question}

REFERENCE ANSWER:
{reference}

CANDIDATE ANSWER:
{candidate}
"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "factuality":   {"type": "integer", "minimum": 1, "maximum": 5},
        "completeness": {"type": "integer", "minimum": 1, "maximum": 5},
        "grounding":    {"type": "integer", "minimum": 1, "maximum": 5},
        "rationale":    {"type": "string"},
    },
    "required": ["factuality", "completeness", "grounding", "rationale"],
}


def llm_judge(question: str, reference: str, candidate: str) -> dict:
    judge = GenerativeModel(GEMINI_PRO_MODEL)
    resp = judge.generate_content(
        JUDGE_PROMPT.format(question=question, reference=reference, candidate=candidate),
        generation_config=GenerationConfig(
            response_mime_type="application/json",
            response_schema=JUDGE_SCHEMA,
            temperature=0.0,
        ),
    )
    return json.loads(resp.text)


# ---------- Eval runner ----------

@dataclass
class Result:
    id: str
    question: str
    candidate: str
    em: float
    rougeL: float
    emb_sim: float
    judge: dict


def run():
    init_vertex()
    corpus = _load_corpus()
    embed_model = TextEmbeddingModel.from_pretrained(EMBEDDING_MODEL)

    results: list[Result] = []
    for ex in GOLDEN:
        print(f"\n--- {ex['id']} ---")
        candidate = system_under_test(ex["question"], corpus)
        r = Result(
            id=ex["id"],
            question=ex["question"],
            candidate=candidate,
            em=exact_match(candidate, ex["reference"]),
            rougeL=rouge_l(candidate, ex["reference"]),
            emb_sim=emb_sim(candidate, ex["reference"], embed_model),
            judge=llm_judge(ex["question"], ex["reference"], candidate),
        )
        results.append(r)
        print(f"  ROUGE-L: {r.rougeL:.3f}  emb_sim: {r.emb_sim:.3f}")
        print(f"  Judge:   {r.judge}")

    # Aggregate
    print("\n=== SUMMARY ===")
    print(f"n={len(results)}")
    print(f"  ROUGE-L avg: {np.mean([r.rougeL for r in results]):.3f}")
    print(f"  emb_sim avg: {np.mean([r.emb_sim for r in results]):.3f}")
    print(f"  judge.factuality avg:   {np.mean([r.judge['factuality'] for r in results]):.2f}")
    print(f"  judge.completeness avg: {np.mean([r.judge['completeness'] for r in results]):.2f}")
    print(f"  judge.grounding avg:    {np.mean([r.judge['grounding'] for r in results]):.2f}")

    # Persist results so you can diff runs
    out = Path(__file__).resolve().parent / "eval_results.json"
    out.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    run()


# ===== Try next =====
# 1. Add five more golden examples — include hard ones (ambiguous, adversarial, multi-hop).
# 2. Replace the naive "concat all docs" corpus with your real RAG from Week 5.
# 3. Add pairwise judging: generate answers from two models and ask the judge which is better.
# 4. Store results in BigQuery keyed by git commit SHA so you can track quality over time.
