"""
LangGraph agent loop — Groq version.

Graph shape:

    START -> agent -> [conditional] -> tool -> agent -> ... -> END
                    \-> END (when the LLM decides it has enough info)

Groq's API is OpenAI-compatible, so the message format and tool-call
format are OpenAI-style, not Anthropic-style. Key differences from the
Anthropic version are called out in comments below.
"""

import json
import operator
from typing import Annotated, TypedDict

from groq import Groq
from langgraph.graph import StateGraph, END

from tools import TOOL_REGISTRY

MAX_ITERATIONS = 5
# Pick a Groq-hosted model that supports tool calling.
# Check https://console.groq.com/docs/models for the current list.
MODEL = "llama-3.3-70b-versatile"

client = Groq()  # reads GROQ_API_KEY from the environment

# OpenAI-style tool definitions: each one is wrapped in {"type": "function", "function": {...}}
# and uses "parameters" instead of Anthropic's "input_schema".
TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_lookup",
            "description": "Get a Wikipedia summary for a topic or entity.",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_retrieve",
            "description": "Search your local document store (ChromaDB) for relevant chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 3},
                },
                "required": ["query"],
            },
        },
    },
]


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    iterations: int
    final_answer: str


def agent_node(state: AgentState) -> AgentState:
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        tools=TOOL_DEFS,
        messages=state["messages"],
    )

    msg = response.choices[0].message

    # Groq/OpenAI puts tool calls in msg.tool_calls, not a stop_reason check.
    # We convert the message object into a plain dict to store in state,
    # since LangGraph state needs to be a plain appendable list of dicts.
    assistant_message = {
        "role": "assistant",
        "content": msg.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in (msg.tool_calls or [])
        ] or None,
    }

    if not msg.tool_calls:
        # No tool call requested -> this is the final answer.
        return {
            "messages": [assistant_message],
            "iterations": state["iterations"] + 1,
            "final_answer": msg.content or "",
        }

    return {
        "messages": [assistant_message],
        "iterations": state["iterations"] + 1,
        "final_answer": "",
    }


def tool_node(state: AgentState) -> AgentState:
    last = state["messages"][-1]
    tool_calls = last.get("tool_calls") or []
    tool_messages = []

    for tc in tool_calls:
        name = tc["function"]["name"]
        # Groq sends arguments as a JSON STRING, not a dict -> must parse it.
        args_dict = json.loads(tc["function"]["arguments"])

        fn, schema_cls = TOOL_REGISTRY[name]
        try:
            args = schema_cls(**args_dict)
            result = fn(args)
        except Exception as e:
            result = f"tool execution error: {e}"

        # OpenAI/Groq format: ONE separate "tool" role message per tool call,
        # matched back to the call via tool_call_id. (Anthropic bundles them
        # into one user message instead -- that's the other big difference.)
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": str(result),
            }
        )

    return {
        "messages": tool_messages,
        "iterations": state["iterations"],
        "final_answer": "",
    }


def should_continue(state: AgentState) -> str:
    if state["final_answer"]:
        return "end"
    if state["iterations"] >= MAX_ITERATIONS:
        return "end"
    return "continue"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tool", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"continue": "tool", "end": END}
    )
    graph.add_edge("tool", "agent")
    return graph.compile()


def run(question: str) -> str:
    app = build_graph()
    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "iterations": 0,
        "final_answer": "",
    }
    final_state = app.invoke(initial_state)
    return final_state["final_answer"] or "Hit max iterations without a final answer."


if __name__ == "__main__":
    q = "What's the population of the capital of the country that won the 2022 FIFA World Cup?"
    print(run(q))
