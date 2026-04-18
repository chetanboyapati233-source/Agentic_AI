"""
02 — Gemini hello world (Weeks 1–2)

Goal:
  - Verify your Vertex AI auth works.
  - Make three calls in three modes: single-shot, streaming, multi-turn chat.
  - See what `temperature` actually does.

Run:
  python 02_gemini_hello_world.py
"""
from _config import init_vertex, GEMINI_MODEL
from vertexai.generative_models import GenerativeModel, GenerationConfig


def single_shot():
    print("\n=== Single-shot ===")
    model = GenerativeModel(GEMINI_MODEL)
    resp = model.generate_content(
        "In one sentence, explain what an LLM token is to a data engineer.",
        generation_config=GenerationConfig(temperature=0.2, max_output_tokens=200),
    )
    print(resp.text)
    print(f"[usage] in={resp.usage_metadata.prompt_token_count} "
          f"out={resp.usage_metadata.candidates_token_count}")


def streaming():
    print("\n=== Streaming ===")
    model = GenerativeModel(GEMINI_MODEL)
    stream = model.generate_content(
        "List 5 differences between batch and streaming data pipelines. "
        "Stream tokens as you go.",
        stream=True,
    )
    for chunk in stream:
        print(chunk.text, end="", flush=True)
    print()


def chat():
    print("\n=== Multi-turn chat ===")
    model = GenerativeModel(
        GEMINI_MODEL,
        system_instruction="You are a senior data engineer who answers concisely.",
    )
    chat_session = model.start_chat()

    for user_msg in [
        "I'm new to GenAI. What's the difference between an embedding and a token?",
        "Give me a one-line code example using Vertex AI for embeddings.",
        "How would I use that vector to search documents?",
    ]:
        print(f"\nUSER: {user_msg}")
        r = chat_session.send_message(user_msg)
        print(f"GEMINI: {r.text}")


def temperature_demo():
    """Show that the same prompt at temperature 0 vs 1 produces different outputs."""
    print("\n=== Temperature demo ===")
    model = GenerativeModel(GEMINI_MODEL)
    prompt = "Give a one-sentence creative tagline for a data quality tool."
    for temp in (0.0, 1.0):
        resp = model.generate_content(
            prompt, generation_config=GenerationConfig(temperature=temp, max_output_tokens=80)
        )
        print(f"  T={temp}: {resp.text.strip()}")


if __name__ == "__main__":
    init_vertex()
    single_shot()
    streaming()
    chat()
    temperature_demo()


# ===== Try next =====
# 1. Try GEMINI_PRO_MODEL ('gemini-1.5-pro-002') and compare quality on the same prompts.
# 2. Add `safety_settings=` and observe what Gemini blocks.
# 3. Time the streaming vs non-streaming versions of the same prompt — first-token latency
#    is what matters for UX, not total time.
