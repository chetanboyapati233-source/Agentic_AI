"""
10 — LangGraph agent: SQL data analyst (Week 8)

Goal:
  - Build an agent as an explicit graph: plan → write SQL → execute → reflect → answer.
  - Add a self-correction loop on SQL errors.
  - Hard-cap iterations so the agent can't spin forever.

Why LangGraph instead of a free-form ReAct loop:
  - Explicit state machine = predictable behavior.
  - Easier to evaluate, debug, and rate-limit.
  - You can add human-in-the-loop checkpoints at specific nodes.

Run:
  python 10_agent_langgraph.py "How many rows in dataset.table?"

Note: this script uses BigQuery if PROJECT_ID is set; otherwise it returns a mock.
"""
from __future__ import annotations
import os
import sys
import operator
from typing import TypedDict, Annotated, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import StateGraph, END

from _config import init_vertex, GEMINI_MODEL, PROJECT_ID, BQ_DATASET


# ---------- Agent state ----------

class State(TypedDict):
    question: str
    schema: str
    sql: str
    rows: str
    error: str
    final: str
    iterations: int


MAX_ITER = 3


# ---------- Helpers ----------

def _get_schema() -> str:
    """Pull a tiny BigQuery schema. Falls back to mock if BQ isn't configured."""
    try:
        from google.cloud import bigquery
        bq = bigquery.Client(project=PROJECT_ID)
        sql = f"""
          SELECT table_name, column_name, data_type
          FROM `{PROJECT_ID}.{BQ_DATASET}.INFORMATION_SCHEMA.COLUMNS`
          ORDER BY table_name, ordinal_position
          LIMIT 50
        """
        rows = list(bq.query(sql).result())
        if rows:
            return "\n".join(f"  {r.table_name}.{r.column_name} ({r.data_type})" for r in rows)
    except Exception:
        pass
    return (
        "  docs.id (STRING)\n"
        "  docs.doc_id (STRING)\n"
        "  docs.text (STRING)\n"
        "  docs.embedding (FLOAT64 ARRAY)"
    )


def _execute_sql(sql: str) -> tuple[str, str]:
    """Returns (rows_str, error_str)."""
    try:
        from google.cloud import bigquery
        bq = bigquery.Client(project=PROJECT_ID)
        job_config = bigquery.QueryJobConfig(maximum_bytes_billed=100 * 1024 * 1024)
        rows = list(bq.query(sql, job_config=job_config).result(max_results=20))
        if not rows:
            return "(no rows)", ""
        header = list(rows[0].keys())
        out = " | ".join(header) + "\n"
        for r in rows:
            out += " | ".join(str(r.get(c)) for c in header) + "\n"
        return out, ""
    except Exception as e:
        return "", str(e)


# ---------- Nodes ----------

llm = None
def get_llm():
    global llm
    if llm is None:
        llm = ChatVertexAI(model_name=GEMINI_MODEL, temperature=0.1)
    return llm


def node_plan(state: State) -> State:
    state["schema"] = _get_schema()
    state["iterations"] = state.get("iterations", 0)
    return state


def node_write_sql(state: State) -> State:
    prompt = f"""You are a SQL expert. Write a single BigQuery SQL query to answer the question.
Use ONLY the schema below. Return ONLY the SQL — no markdown, no commentary.

SCHEMA:
{state['schema']}

QUESTION: {state['question']}
"""
    if state.get("error"):
        prompt += f"\n\nThe previous SQL failed with this error. Fix it:\n{state['error']}\nPrevious SQL:\n{state['sql']}"
    msg = get_llm().invoke([HumanMessage(content=prompt)])
    sql = msg.content.strip().strip("`")
    if sql.lower().startswith("sql"):
        sql = sql[3:].strip()
    state["sql"] = sql
    state["iterations"] += 1
    return state


def node_execute(state: State) -> State:
    rows, err = _execute_sql(state["sql"])
    state["rows"] = rows
    state["error"] = err
    return state


def node_summarize(state: State) -> State:
    prompt = f"""Summarize the SQL result for the user.

Question: {state['question']}
SQL: {state['sql']}
Result:
{state['rows']}

Be concise (2-3 sentences). Mention the actual values."""
    msg = get_llm().invoke([HumanMessage(content=prompt)])
    state["final"] = msg.content
    return state


def route_after_execute(state: State) -> str:
    if not state["error"]:
        return "summarize"
    if state["iterations"] >= MAX_ITER:
        # Give up gracefully
        state["final"] = (
            f"I tried {MAX_ITER} times but couldn't write a working query. "
            f"Last error: {state['error']}"
        )
        return END
    return "write_sql"


# ---------- Build the graph ----------

def build_graph():
    g = StateGraph(State)
    g.add_node("plan", node_plan)
    g.add_node("write_sql", node_write_sql)
    g.add_node("execute", node_execute)
    g.add_node("summarize", node_summarize)

    g.set_entry_point("plan")
    g.add_edge("plan", "write_sql")
    g.add_edge("write_sql", "execute")
    g.add_conditional_edges("execute", route_after_execute,
                            {"write_sql": "write_sql", "summarize": "summarize", END: END})
    g.add_edge("summarize", END)
    return g.compile()


if __name__ == "__main__":
    init_vertex()
    question = sys.argv[1] if len(sys.argv) > 1 else "How many distinct doc_ids are in the docs table?"
    app = build_graph()
    final_state = app.invoke({"question": question})

    print(f"\nQuestion:  {question}")
    print(f"Schema:\n{final_state['schema']}")
    print(f"SQL:       {final_state['sql']}")
    print(f"Rows:\n{final_state['rows']}")
    print(f"\nAnswer:\n{final_state.get('final', '(none)')}")


# ===== Try next =====
# 1. Add a "validator" node before execute that checks for DELETE/UPDATE/CREATE — refuse those.
# 2. Add a human-in-the-loop interrupt: pause after node_write_sql for user approval before exec.
# 3. Replace the schema fetch with a RAG retrieval over BQ INFORMATION_SCHEMA + docs.
