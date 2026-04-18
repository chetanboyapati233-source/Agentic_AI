import sqlite3
from langchain_core.tools import tool
from vector_store import search_documents as vs_search

# @tool decorator replaces the manual TOOLS JSON definition we wrote in company/agent.py
# LangChain reads the function name + docstring to describe the tool to the LLM

@tool
def execute_sql(query: str) -> list:
    """Run a SQLite SQL query on the company database.
    Use this for questions about employees, salaries, departments, and sales data.
    Schema: employees(id, name, department, salary, hire_date), sales(id, employee_id, amount, sale_date)"""
    conn = sqlite3.connect("company.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        cur.execute(query)
        rows = [dict(r) for r in cur.fetchall()]
    except Exception as e:
        rows = [{"error": str(e)}]
    finally:
        conn.close()
    return rows

@tool
def search_documents(query: str) -> str:
    """Search company HR policies and handbooks.
    Use this for questions about leave days, working hours, commission rates,
    sales targets, performance reviews, expense reimbursement, or any company policy."""
    return vs_search(query)
