# RAG patterns cheat sheet

Retrieval-Augmented Generation (RAG) is the pattern of augmenting an LLM's prompt with
documents you retrieve from your own data at query time. It is the workhorse of applied
GenAI: no fine-tuning needed, facts stay fresh, and you can cite sources.

## The canonical pipeline

Ingest phase (offline, batch):

1. Source documents from Cloud Storage, BigQuery, wikis, tickets, etc.
2. Parse (e.g., Document AI for PDFs), clean HTML, strip boilerplate.
3. Chunk into passages of a few hundred words with light overlap.
4. Embed each chunk with a text embedding model.
5. Upsert into a vector index with metadata (source ID, date, tags, tenant).

Query phase (online, low latency):

1. User asks a question.
2. Embed the question.
3. Retrieve top-K nearest chunks (optionally filter by metadata).
4. Compose a prompt: "Answer using only the context below. Cite sources. Say 'I don't know'
   if the context is insufficient."
5. Send to the LLM. Return the answer and the source IDs.

## Quality knobs

- **Chunk size and overlap.** Too small → lose context. Too big → too few chunks fit in
  the prompt. Typical starting point: 500–800 tokens, 10–15% overlap.
- **Top-K.** Higher K = more context but also more noise. 3–5 is a reasonable default
  once you add reranking.
- **Hybrid search.** Combine semantic (vector) with lexical (BM25 / full-text) for SKUs,
  codes, names, and other exact-match tokens.
- **Reranking.** Put a cross-encoder reranker after retrieval. Fetch 30–50 candidates,
  rerank to top 3–5. This alone often improves recall@K by 5–15 points.
- **Query rewriting.** Rewrite the user query before embedding (e.g., expand acronyms,
  add context from chat history). HyDE — generate a hypothetical answer and embed that —
  helps for out-of-distribution queries.

## When RAG fails

- **Ambiguous queries** get mis-retrieved. Add disambiguation prompts or clarifying
  questions before retrieval.
- **Stale indexes.** If your ingest lags by a day, recency queries break. Show the
  freshness date to the user.
- **Hallucinated citations** are the worst failure mode — the model fabricates source
  IDs. Mitigate by validating IDs post-generation and refusing to answer if sources
  don't match retrieved chunks.
- **Cross-document reasoning.** RAG surfaces chunks in isolation. If the answer requires
  synthesizing three sources, rephrase the prompt to instruct that and increase K.

## When NOT to use RAG

- The answer is deterministic and lives in a structured source. Query BigQuery or a
  database; don't ask an LLM to summarize it.
- The task is creative writing, not factual lookup.
- Your domain vocabulary is nonstandard enough that prompting fails — fine-tune instead.
