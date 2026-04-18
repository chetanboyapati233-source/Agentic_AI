"""
03 — Prompt engineering (Week 3)

Goal:
  - Feel the quality delta between zero-shot, few-shot, and chain-of-thought prompts.
  - Force strict JSON output with response_schema — don't parse natural language.

Run:
  python 03_prompt_engineering.py
"""
import json
from _config import init_vertex, GEMINI_MODEL
from vertexai.generative_models import GenerativeModel, GenerationConfig


model = None


def get_model():
    global model
    if model is None:
        model = GenerativeModel(GEMINI_MODEL)
    return model


# ---------- 1. Sentiment classification: zero-shot vs few-shot ----------

REVIEWS = [
    "The pipeline finally works, but it took three weeks too long.",
    "Absolutely love the new orchestrator, my life is easier.",
    "Meh. It does the job.",
    "I cannot believe this is the state of the art.",  # ambiguous
]


def zero_shot_sentiment(review: str) -> str:
    prompt = f"Classify the sentiment of this review as positive, negative, or neutral:\n{review}"
    return get_model().generate_content(prompt).text.strip()


def few_shot_sentiment(review: str) -> str:
    prompt = f"""Classify the sentiment of the REVIEW as positive, negative, or neutral.
Return ONLY the label.

Examples:
REVIEW: "The dashboards are beautiful!" → positive
REVIEW: "Crashes every Monday." → negative
REVIEW: "It exists." → neutral
REVIEW: "Does what it says on the tin." → neutral
REVIEW: "Infuriatingly slow at scale." → negative

REVIEW: "{review}" →"""
    return get_model().generate_content(prompt).text.strip()


# ---------- 2. Chain-of-thought on a tricky math question ----------

def cot_math() -> str:
    prompt = """A data pipeline runs every 15 minutes. Each run reads 200 MB and
writes 50 MB back. How many GB are read and written in a 24-hour day?
Think step by step before giving the final numbers."""
    return get_model().generate_content(prompt).text


# ---------- 3. Structured JSON output with a schema ----------

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "product": {"type": "string"},
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "reasons": {"type": "array", "items": {"type": "string"}},
        "severity_1_to_5": {"type": "integer"},
    },
    "required": ["product", "sentiment", "severity_1_to_5"],
}


def extract_structured(review: str) -> dict:
    prompt = (
        "Extract structured information from the customer review below. "
        "Severity 1 = minor, 5 = blocking.\n\nREVIEW: " + review
    )
    resp = get_model().generate_content(
        prompt,
        generation_config=GenerationConfig(
            response_mime_type="application/json",
            response_schema=EXTRACT_SCHEMA,
            temperature=0.0,
        ),
    )
    return json.loads(resp.text)


# ---------- 4. A two-step prompt chain ----------
# Pattern: stage 1 extracts → stage 2 writes. Each has a narrow job.

def draft_incident_email(review: str) -> str:
    extracted = extract_structured(review)
    prompt = f"""You are writing a short internal incident email (3 sentences max)
based on the extracted signal below.

{json.dumps(extracted, indent=2)}

Tone: calm, factual, action-oriented. Include severity and the top reason."""
    return get_model().generate_content(prompt).text


if __name__ == "__main__":
    init_vertex()

    print("=== 1. Zero-shot vs few-shot sentiment ===")
    for r in REVIEWS:
        print(f"\n  {r!r}")
        print(f"    zero-shot → {zero_shot_sentiment(r)}")
        print(f"    few-shot  → {few_shot_sentiment(r)}")

    print("\n=== 2. Chain-of-thought math ===")
    print(cot_math())

    print("\n=== 3. Structured JSON extract ===")
    review = "The new Airflow upgrade broke half our DAGs. Critical pipelines are down."
    print(json.dumps(extract_structured(review), indent=2))

    print("\n=== 4. Prompt chain: extract → draft email ===")
    print(draft_incident_email(review))


# ===== Try next =====
# 1. Add three more examples to few-shot — ones that look like edge cases from your domain.
# 2. Break the schema: pass an ambiguous review and watch Gemini refuse to produce garbage fields.
# 3. Rewrite CoT with a "final_answer" JSON block at the end so code can parse it reliably.
