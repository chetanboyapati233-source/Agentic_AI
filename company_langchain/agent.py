import os
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain.agents import create_agent
from tools import execute_sql, search_documents

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=os.environ["GROQ_API_KEY"]
)

tools = [execute_sql, search_documents]

# LangGraph's create_react_agent replaces the old AgentExecutor
SYSTEM_PROMPT = """You are a company data assistant. You MUST always use a tool to answer every question. Never answer directly from memory.

Rules:
- ALWAYS call execute_sql for ANY question about employees, salaries, departments, sales, or database data
- ALWAYS call search_documents for ANY question about policies, leave, working hours, commission, benefits
- NEVER write SQL in your response text — always call the execute_sql tool instead
- NEVER explain how to do something — just call the right tool and return the result"""

agent = create_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM_PROMPT
)

def run_agent(question: str) -> dict:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    tool_used = None
    tool_input = None
    tool_result = None

    # LangGraph returns all messages — find the tool call and result
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_used = msg.tool_calls[0]["name"]
            tool_input = msg.tool_calls[0]["args"]
        if hasattr(msg, "name") and msg.name in ("execute_sql", "search_documents"):
            tool_result = msg.content

    answer = result["messages"][-1].content

    return {
        "answer": answer,
        "tool_used": tool_used,
        "tool_input": tool_input,
        "tool_result": tool_result
    }
