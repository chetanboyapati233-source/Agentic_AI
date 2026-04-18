# GenAI for Data Engineers — Starter Kit

A complete, GCP-first learning pack to take you from data engineer to AI engineer over
about 12 weeks of evenings (5–8 hrs/week). Built for you, Chetan.

---

## What's in this folder

```
outputs/
├── GenAI_AI_Engineer_Roadmap.docx   ← 12-week study plan, week by week
├── Interview_Prep_Kit.docx          ← 47 Q&A, 5 system designs, behavioral prep
└── projects/                        ← 12 runnable Vertex AI labs + sample data
    ├── README.md
    ├── requirements.txt
    ├── .env.example
    ├── _config.py
    ├── 01_setup_auth.md
    ├── 02_gemini_hello_world.py
    ├── 03_prompt_engineering.py
    ├── 04_embeddings_basics.py
    ├── 05_chunking.py
    ├── 06_rag_inmemory.py
    ├── 07_rag_vertex_vector_search.py
    ├── 08_rag_bigquery_vector.py
    ├── 09_function_calling.py
    ├── 10_agent_langgraph.py
    ├── 11_evaluation.py
    ├── 12_finetuning_supervised.py
    ├── 13_capstone_de_assistant.py
    └── sample_docs/  (four short reference docs used by the RAG labs)
```

---

## Suggested first 30 minutes

1. **Open the roadmap.** Read the "How to use this roadmap" and "Roadmap at a glance" sections. Those two together are less than 2 pages.
2. **Skim the Interview Prep Kit's Section 1 — LLM basics.** Five minutes. See what you already know; that tells you how fast to move through Weeks 1–2.
3. **Run `01_setup_auth.md`.** Get your GCP project, ADC, and a working Python env squared away before Week 1 officially starts. This is the boring part — do it when you're fresh so it doesn't eat into your real learning time.

---

## The 12-week arc

| Phase | Weeks | What you'll own by the end |
|------|------|----------------------------|
| Foundations | 1–3 | Can call Gemini from Python, understands tokens, prompt patterns |
| RAG | 4–6 | Can build a production-grade RAG system on Vertex AI |
| Agents | 7–8 | Can wire function-calling + a multi-step LangGraph agent |
| Production | 9–11 | Can evaluate, fine-tune, and ship safely |
| Capstone | 12 | Data Engineer Assistant demo + interview-ready portfolio |

Each week in the roadmap has: learning objectives, core concepts, GCP services to use,
a hands-on lab (mapped to the scripts above), three curated readings, and three self-check
questions.

---

## How to pair the three deliverables

- **Weekdays**: read one section of the roadmap → run the matching lab → edit the "Try next" items at the bottom.
- **Weekend**: pick 5 Q&A from the Interview Prep Kit and answer them out loud before checking the answers. Revisit any you stumbled on.
- **Week 10 onward**: start doing 1 system-design scenario per week on a whiteboard. Write down your answer, then compare to the proposed architecture.
- **Week 12**: your capstone (`projects/13_capstone_de_assistant.py`) becomes the project you walk through in every interview.

---

## Notes on cost

Everything is designed to run cheap. The two things that will run up a bill if you forget
about them:
- **Vertex AI Vector Search deployed endpoint** (Week 5–6) — billed hourly while running. Run `python 07_rag_vertex_vector_search.py cleanup` at end of day.
- **Fine-tuning job** (Week 10) — single-digit to low-double-digit dollars per run. Don't trigger it casually.

Set a billing alert at $10 before starting Week 5. Set one at $50 before Week 10.

---

## If something doesn't work

The scripts are starting points, not production code. Common gotchas are called out in
the script comments and in the roadmap's week-1 setup section. Model names (`gemini-1.5-flash-002`,
`text-embedding-004`) update occasionally — check the Vertex AI docs if the SDK complains
about a model ID.

Good luck. Ship the capstone.
