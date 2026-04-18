import streamlit as st
import os
from agent import run_agent

st.set_page_config(page_title="Company AI Assistant", page_icon="🤖", layout="wide")

st.title("Company AI Assistant")
st.caption("Ask anything about employees, sales data, or company policies")

with st.sidebar:
    st.header("What can I ask?")
    st.markdown("**Database questions:**")
    st.markdown("- Who are the top earners in Engineering?")
    st.markdown("- What is the total sales in 2024?")
    st.markdown("- Which employee made the most sales?")
    st.markdown("- List all employees in HR")
    st.markdown("---")
    st.markdown("**Policy questions:**")
    st.markdown("- How many leave days do I get?")
    st.markdown("- What is the commission structure?")
    st.markdown("- What are the working hours?")
    st.markdown("- What is the sales target per quarter?")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("tool_used"):
            with st.expander(f"Tool used: `{msg['tool_used']}`"):
                st.write("**Input:**")
                st.code(str(msg.get("tool_input", "")))
                st.write("**Raw Result:**")
                st.write(msg.get("tool_result", ""))

if question := st.chat_input("Ask a question about employees, sales, or company policy..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = run_agent(question)

        st.write(result["answer"])

        if result["tool_used"]:
            with st.expander(f"Tool used: `{result['tool_used']}`"):
                st.write("**Input:**")
                st.code(str(result.get("tool_input", "")))
                st.write("**Raw Result:**")
                st.write(result.get("tool_result", ""))

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "tool_used": result["tool_used"],
        "tool_input": result["tool_input"],
        "tool_result": result["tool_result"]
    })
