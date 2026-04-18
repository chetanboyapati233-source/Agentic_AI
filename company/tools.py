import sqlite3
from vector_store import search_documents as vs_search

def execute_sql(query: str) -> list:
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

def search_documents(query: str) -> str:
    return vs_search(query)
