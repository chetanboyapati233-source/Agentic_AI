# GenAI for Data Engineers — Project Pack

Companion labs to `GenAI_AI_Engineer_Roadmap.docx`. Every week in the roadmap has a matching
script in this folder. Run them in order; each script depends on concepts (not code) from the
previous.

Everything uses **Vertex AI + Gemini on GCP**, so your GCP data-engineering skills carry over.

---

## Quick start

```bash
# 1. Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install deps
pip install -r requirements.txt

# 3. Authenticate to GCP
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

# 4. Enable required APIs (one time)
gcloud services enable aiplatform.googleapis.com \
                       bigquery.googleapis.com \
                       storage.googleapis.com

# 5. Configure env
cp .env.example .env
# Edit .env and set PROJECT_ID, LOCATION, etc.

# 6. Smoke test
python 02_gemini_hello_world.py
```

---

## Files, mapped to the roadmap

| Week | Topic | File |
|------|-------|------|
| 1–2 | Setup & first Gemini call | `01_setup_auth.md` + `02_gemini_hello_world.py` |
| 3 | Prompt engineering | `03_prompt_engineering.py` |
| 4 | Embeddings & similarity | `04_embeddings_basics.py` |
| 5 | Chunking strategies | `05_chunking.py` |
| 5 | In-memory RAG | `06_rag_inmemory.py` |
| 5–6 | RAG with Vertex Vector Search | `07_rag_vertex_vector_search.py` |
| 5–6 | RAG with BigQuery vector | `08_rag_bigquery_vector.py` |
| 7 | Function calling | `09_function_calling.py` |
| 8 | LangGraph agent | `10_agent_langgraph.py` |
| 9 | Evaluation harness | `11_evaluation.py` |
| 10 | Supervised fine-tuning | `12_finetuning_supervised.py` |
| 12 | Capstone: Data Engineer Assistant | `13_capstone_de_assistant.py` |

`sample_docs/` holds a handful of short docs used by the RAG labs so you can run end-to-end
without hunting for data.

---

## What will cost money

Everything in this pack is designed to run cheap (well under $5 if you don't leave loops running),
but you should know what is billable:

- **Gemini calls** (`02`–`13`): a few cents per run on Flash; single-digit dollars on Pro if you abuse it.
- **Embedding calls** (`04`–`08`): fractions of a cent per 1k tokens.
- **Vertex AI Vector Search index** (`07`): has an hourly serving cost while a deployed index endpoint exists. **Delete the index endpoint when you're done each session.** Don't leave it running overnight.
- **BigQuery** (`08`, `13`): query-scan cost; sample datasets are small.
- **Fine-tuning** (`12`): single-digit to low-double-digit dollars per job. Read the billing page before running.

Set a **billing budget** on your project with an alert at $10 before starting Week 5.

---

## How to use each lab

Each script is runnable standalone (`python NN_name.py`) and includes:

- **Docstring** at the top explaining the learning goal.
- **TODOs** you should edit (try different prompts, change params) — the learning comes from poking.
- **"Try next"** section at the bottom: small challenges to extend the script.

Treat them as starting points, not finished products. The roadmap's self-check questions are
answered by actually modifying these files and seeing what happens.
