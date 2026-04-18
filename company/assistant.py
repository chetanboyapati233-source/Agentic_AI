import sqlite3
import os
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"

SCHEMA = """
Tables:
- employees(id, name, department, salary, hire_date)
- sales(id, employee_id, amount, sale_date)
"""

def get_sql_from_question(question: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": f"You are a SQL expert. Given a question, return ONLY valid SQLite SQL. No explanation. No markdown. Just the SQL query.\nSchema: {SCHEMA}"
            },
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content.strip()

def run_query(sql: str) -> list:
    conn = sqlite3.connect("company.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(sql)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def summarize_results(question: str, sql: str, results: list) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": f"Question: {question}\nSQL used: {sql}\nResults: {results}\n\nAnswer the question in plain English in 1-2 sentences."
            }
        ]
    )
    return response.choices[0].message.content.strip()

def ask(question: str):
    print(f"\n{'='*50}")
    print(f"Question: {question}")
    sql = get_sql_from_question(question)
    print(f"SQL:      {sql}")
    results = run_query(sql)
    print(f"Results:  {results}")
    answer = summarize_results(question, sql, results)
    print(f"Answer:   {answer}")
    print('='*50)

if __name__ == "__main__":
    ask("Who are the top earners in Engineering?")
    ask("What is the total sales amount in 2024?")
    ask("Which employee made the most sales?")
