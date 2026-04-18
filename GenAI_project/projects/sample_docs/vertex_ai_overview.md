# Vertex AI — a practical overview for data engineers

Vertex AI is Google Cloud's unified platform for building, deploying, and operating ML and
generative AI workloads. For a data engineer already comfortable with BigQuery, Cloud Storage,
and Dataflow, Vertex AI is the natural place to extend your skills into GenAI: the same IAM,
the same networking, the same logging — with a purpose-built surface for models.

## Core surfaces

Vertex AI is not a single product. It's several sub-services that share infrastructure:

1. **Model Garden** — a catalog of foundation models: Google's Gemini family, partner models
   (Anthropic Claude, Meta Llama, Mistral), and open-source options. You browse, compare,
   and deploy from here.
2. **Vertex AI Studio** — a browser-based playground. Paste a prompt, toggle parameters,
   compare models side by side, then export the prompt as code. First stop for any new task.
3. **Gemini API** — the production endpoint. Call it from the `vertexai` Python SDK,
   `google.generativeai`, the REST API, or via LangChain's integration.
4. **Embeddings API** — turn text (or images, or video) into fixed-size vectors with
   `text-embedding-004` and multimodal variants.
5. **Vector Search** (formerly Matching Engine) — a managed approximate nearest neighbor
   index. Scales from thousands to billions of vectors.
6. **Agent Builder** — higher-level managed agents with grounding on your data sources
   (Drive, websites, Cloud Storage, BigQuery).
7. **Reasoning Engine** — managed runtime for code-first agents written with LangChain,
   LangGraph, or the Vertex AI Agent SDK.
8. **Gen AI Evaluation Service** — pointwise and pairwise evaluation with prebuilt metrics
   and LLM-as-judge support.
9. **Pipelines** — Kubeflow-based orchestration for data-prep → train/tune → eval → deploy.
10. **Safety filters & Model Armor** — input/output filtering, prompt injection defense,
    PII redaction when combined with Cloud DLP.

## Why data engineers have an advantage

Production GenAI systems are data systems first:

- **Ingest and normalize** source content (documents, wiki, logs) into chunks.
- **Embed and index** those chunks in a vector store.
- **Retrieve** at query time with filtering on metadata (tenant, permissions, recency).
- **Observe** every request: tokens, latency, cost, quality signals.
- **Govern** access with IAM, VPC-SC, and audit logs.

Every one of those is a pipeline problem, and pipelines are what you already build. The
delta from data engineer to AI engineer is the reasoning layer — how the retrieved context
is turned into an answer — plus a few new failure modes (hallucination, prompt injection,
cost runaway) that you need to plan for.

## Choosing between managed and code-first

Vertex AI offers both. The rule of thumb:

- **Managed** (Agent Builder, out-of-the-box RAG) is faster to stand up but gives you less
  control over retrieval strategy, prompts, and eval. Great for internal help-desks or
  first prototypes.
- **Code-first** (Gemini SDK + your own retrieval) gives you every knob. Most production
  AI engineers live here. It is also what interviewers will ask you about.

Pick managed when the task is standard (employee Q&A over Confluence) and code-first when
the task is distinctive to your business (SQL assistant over your schema, support bot
with specific escalation rules).

## Cost shape

Gemini is priced per token, input and output separately. Output is more expensive than
input. Flash is roughly 5–10× cheaper than Pro at the time of writing. Long context (>32k
tokens) has a premium tier.

Three cost levers you will actually use:

1. **Route to the cheapest model that works** — Flash for simple classification,
   Pro for complex reasoning.
2. **Cap `max_output_tokens`** — unbounded generation is the most common overrun.
3. **Cache prompts** — Vertex AI supports context caching for repeated system
   prompts or large shared contexts.
