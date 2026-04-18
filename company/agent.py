import json
import os
from groq import Groq
from tools import execute_sql, search_documents

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

SCHEMA = """
Tables:
- employees(id, name, department, salary, hire_date)
- sales(id, employee_id, amount, sale_date)
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": "Run a SQL query on the company database. Use this for questions about employees, salaries, departments, sales numbers, and any quantitative data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": f"A valid SQLite SQL query. Schema: {SCHEMA}"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search company documents and HR policies. Use this for questions about leave, benefits, working hours, commission structure, sales targets, code of conduct, or any company policy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant policy or handbook content"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

def run_agent(user_question: str) -> dict:
    messages = [
        {
            "role": "system",
            "content": f"""You are a helpful company assistant with access to two tools:
1. execute_sql — for employee data, salaries, sales numbers from the database
2. search_documents — for HR policies, leave rules, commission structure, company guidelines

Always pick the right tool. Database schema: {SCHEMA}"""
        },
        {"role": "user", "content": user_question}
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto"
    )

    message = response.choices[0].message
    tool_used = None
    tool_input = None
    tool_result = None

    if message.tool_calls:
        tool_call = message.tool_calls[0]
        tool_used = tool_call.function.name
        tool_input = json.loads(tool_call.function.arguments)

        if tool_used == "execute_sql":
            tool_result = execute_sql(tool_input["query"])
        elif tool_used == "search_documents":
            tool_result = search_documents(tool_input["query"])

        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [tc.model_dump() for tc in message.tool_calls]
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(tool_result)
        })

        final = client.chat.completions.create(model=MODEL, messages=messages)
        answer = final.choices[0].message.content
    else:
        answer = message.content

    return {
        "answer": answer,
        "tool_used": tool_used,
        "tool_input": tool_input,
        "tool_result": tool_result
    }
