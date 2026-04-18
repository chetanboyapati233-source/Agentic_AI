# Gemini cost and latency guide

LLM cost is priced per token. On Vertex AI, input tokens and output tokens are billed at
different rates, and output is always more expensive. Understanding this is the
difference between a $20/month side project and a $20,000 surprise.

## Token accounting

1 token ≈ 0.75 English words. A 4-line prompt is around 50–80 tokens. A 10-page PDF is
roughly 3,000–5,000 tokens. Your RAG context might be 5,000–20,000 tokens on a big query.

Check token counts before sending:

```python
from vertexai.generative_models import GenerativeModel
model = GenerativeModel("gemini-1.5-flash-002")
print(model.count_tokens("Your prompt here"))
```

## Flash vs Pro

- **Gemini 1.5 Flash** — fast, cheap, great for classification, extraction, simple RAG.
  Roughly 5–10× cheaper than Pro.
- **Gemini 1.5 Pro** — better reasoning, longer output, stronger on hard tasks. Use it
  where Flash fails, not as a default.
- **Router pattern.** A common production layout: a cheap model classifies the query,
  then routes to Flash or Pro based on difficulty.

## Four ways to cut cost fast

1. **Cap `max_output_tokens`.** Runaway generation is the most common bill spike.
2. **Use JSON schemas and structured output.** Shorter, more predictable outputs.
3. **Prompt caching.** If the same 10k-token system prompt runs on every call, cache it.
4. **Batch where latency permits.** For offline jobs, batch embeddings in 250-item chunks.

## Latency budgets

- First-token latency matters more than total latency for anything user-facing. Always
  stream.
- Agents with 5+ tool calls will feel slow no matter what. Consider showing progress
  updates between steps.
- Vector retrieval is typically 50–200 ms on Vertex AI Vector Search for a well-tuned
  index.
