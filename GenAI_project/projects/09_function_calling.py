"""
09 — Function calling / tool use (Week 7)

Goal:
  - Declare tools (name + description + JSON schema).
  - Let Gemini pick a tool, execute it, feed results back, get a final answer.
  - See the full call → response → follow-up loop.

Run:
  python 09_function_calling.py
"""
from __future__ import annotations
import datetime as dt
import json
import random

from vertexai.generative_models import (
    GenerativeModel, Tool, FunctionDeclaration, Part, GenerationConfig,
)

from _config import init_vertex, GEMINI_MODEL, PROJECT_ID


# ---------- Tool implementations (what actually runs) ----------

def get_time(timezone: str = "UTC") -> dict:
    now = dt.datetime.now(dt.timezone.utc)
    return {"iso": now.isoformat(), "timezone": timezone}


def get_weather(city: str) -> dict:
    # Mock — in real life you'd call a weather API here.
    return {
        "city": city,
        "temp_c": round(random.uniform(-5, 35), 1),
        "conditions": random.choice(["sunny", "rainy", "cloudy", "snowy"]),
    }


def query_bigquery_preview(sql: str) -> dict:
    """
    Read-only BigQuery preview. Deliberately simple — we reject anything that isn't
    a SELECT and cap rows. In production you'd also add cost caps and a scan limit.
    """
    stripped = sql.strip().rstrip(";").lower()
    if not stripped.startswith("select"):
        return {"error": "Only SELECT queries are allowed."}
    # Lazy import so this script runs without bigquery set up
    try:
        from google.cloud import bigquery
        bq = bigquery.Client(project=PROJECT_ID)
        job_config = bigquery.QueryJobConfig(
            dry_run=False,
            maximum_bytes_billed=100 * 1024 * 1024,  # 100 MB cap
        )
        rows = list(bq.query(sql, job_config=job_config).result(max_results=10))
        return {"rows": [dict(r) for r in rows]}
    except Exception as e:
        return {"error": str(e)}


TOOL_IMPL = {
    "get_time": get_time,
    "get_weather": get_weather,
    "query_bigquery_preview": query_bigquery_preview,
}


# ---------- Tool declarations (what the model sees) ----------

tools = Tool(function_declarations=[
    FunctionDeclaration(
        name="get_time",
        description="Returns the current UTC time. Use when the user asks for the time or date.",
        parameters={
            "type": "object",
            "properties": {"timezone": {"type": "string", "description": "IANA timezone, default UTC"}},
        },
    ),
    FunctionDeclaration(
        name="get_weather",
        description="Returns current mock weather for a city. Use for weather questions.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    ),
    FunctionDeclaration(
        name="query_bigquery_preview",
        description=(
            "Runs a READ-ONLY SELECT query against BigQuery and returns up to 10 rows. "
            "Only use this if the user explicitly wants data from BigQuery."
        ),
        parameters={
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "A single SELECT statement"}},
            "required": ["sql"],
        },
    ),
])


SYSTEM = """You are a helpful data engineer assistant.
Prefer calling a tool over guessing. Only call query_bigquery_preview when the user
clearly asks about BigQuery data. Never invent column names; ask the user first.
When you return a final answer, include the tool outputs you used."""


def chat(user_prompts: list[str]):
    model = GenerativeModel(GEMINI_MODEL, system_instruction=SYSTEM, tools=[tools])
    session = model.start_chat()

    for prompt in user_prompts:
        print(f"\nUSER: {prompt}")
        response = session.send_message(prompt)

        # Loop: while the model asks for a tool, run it and send the result back.
        while True:
            parts = response.candidates[0].content.parts
            fn_calls = [p.function_call for p in parts if p.function_call]
            if not fn_calls:
                break
            tool_responses = []
            for fc in fn_calls:
                name = fc.name
                args = dict(fc.args or {})
                print(f"  → TOOL CALL: {name}({args})")
                try:
                    result = TOOL_IMPL[name](**args)
                except Exception as e:
                    result = {"error": str(e)}
                print(f"    result: {json.dumps(result)[:200]}")
                tool_responses.append(Part.from_function_response(name=name, response=result))
            response = session.send_message(tool_responses)

        print(f"MODEL: {response.text}")


if __name__ == "__main__":
    init_vertex()
    chat([
        "What time is it?",
        "And what's the weather in Bengaluru?",
        # This one will (hopefully) refuse without a real schema — that's the right move.
        "Also tell me how many rows are in the orders table in BigQuery.",
    ])


# ===== Try next =====
# 1. Tighten the system prompt to make the model ask "which project/dataset?" before
#    running BQ. Observe how tool use drops.
# 2. Add a `send_email` tool and use `tool_config` to ban it from being auto-selected
#    (force the model to ask permission).
# 3. Parallel tool calls: ask a question that needs both get_time and get_weather at once.
