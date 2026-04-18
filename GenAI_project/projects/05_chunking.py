"""
05 — Chunking strategies (Week 5)

Goal:
  - Implement fixed-size, sentence-aware, and recursive chunkers.
  - See how chunk boundaries affect retrievable meaning.

Key intuition:
  A bad chunk strategy is the #1 cause of bad RAG. If your chunks cut off mid-idea,
  retrieval surfaces fragments the LLM can't reason about.

Run:
  python 05_chunking.py
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path


SAMPLE = (Path(__file__).resolve().parent / "sample_docs" / "vertex_ai_overview.md")


@dataclass
class Chunk:
    text: str
    start: int
    end: int
    meta: dict


# ---------- Strategy 1: fixed-size with overlap ----------

def chunk_fixed(text: str, size: int = 500, overlap: int = 80) -> list[Chunk]:
    chunks = []
    i = 0
    while i < len(text):
        end = min(i + size, len(text))
        chunks.append(Chunk(text[i:end], i, end, {"strategy": "fixed"}))
        if end == len(text):
            break
        i = end - overlap
    return chunks


# ---------- Strategy 2: sentence-aware ----------
# Respect sentence boundaries so we don't split mid-sentence.
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_sentences(text: str, target_size: int = 500) -> list[Chunk]:
    sentences = _SENT_RE.split(text)
    chunks, buf, buf_start, cursor = [], [], 0, 0
    for s in sentences:
        if sum(len(x) for x in buf) + len(s) > target_size and buf:
            chunk_text = " ".join(buf)
            chunks.append(Chunk(chunk_text, buf_start, buf_start + len(chunk_text),
                                {"strategy": "sentences"}))
            buf, buf_start = [], cursor
        buf.append(s)
        cursor += len(s) + 1
    if buf:
        chunk_text = " ".join(buf)
        chunks.append(Chunk(chunk_text, buf_start, buf_start + len(chunk_text),
                            {"strategy": "sentences"}))
    return chunks


# ---------- Strategy 3: recursive by markdown structure ----------
# Prefer splitting on H2 > paragraph > sentence. This keeps logical sections intact.

def chunk_markdown(text: str, target_size: int = 800) -> list[Chunk]:
    sections = re.split(r"\n(?=##\s)", text)  # split at H2 boundaries
    chunks = []
    cursor = 0
    for sec in sections:
        if len(sec) <= target_size:
            chunks.append(Chunk(sec.strip(), cursor, cursor + len(sec),
                                {"strategy": "markdown"}))
        else:
            # Fall back to sentence chunking within oversized sections
            for c in chunk_sentences(sec, target_size):
                c.meta["strategy"] = "markdown→sentences"
                c.start += cursor
                c.end += cursor
                chunks.append(c)
        cursor += len(sec) + 1
    return chunks


def _brief(c: Chunk) -> str:
    head = c.text[:80].replace("\n", " ")
    return f"[{c.meta['strategy']:<20}] {len(c.text):>4} chars — {head}..."


if __name__ == "__main__":
    if not SAMPLE.exists():
        print(f"Missing: {SAMPLE}. Run from the projects/ folder where sample_docs/ lives.")
        raise SystemExit(1)
    text = SAMPLE.read_text()
    print(f"Source: {SAMPLE.name}  ({len(text)} chars)\n")

    for name, chunks in [
        ("FIXED (500/80)",      chunk_fixed(text, 500, 80)),
        ("SENTENCES (~500)",    chunk_sentences(text, 500)),
        ("MARKDOWN (~800)",     chunk_markdown(text, 800)),
    ]:
        print(f"=== {name} — {len(chunks)} chunks ===")
        for c in chunks[:6]:
            print("  " + _brief(c))
        if len(chunks) > 6:
            print(f"  ... +{len(chunks) - 6} more")
        print()


# ===== Try next =====
# 1. Add a "parent document retrieval" variant: index small chunks, return the parent section
#    (larger, richer context) when the small chunk is a top hit.
# 2. Replace the hand-rolled splitter with LangChain's RecursiveCharacterTextSplitter and
#    compare chunk boundaries.
# 3. Add an "overlap context" footer to each chunk (e.g., last sentence of previous chunk),
#    so retrieved chunks don't lose reference to their predecessor.
