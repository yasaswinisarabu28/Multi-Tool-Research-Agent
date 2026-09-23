# Multi-Tool Research Agent — Scaffold

## Files
- `schemas.py` — Pydantic input schemas per tool (reused as MCP schemas in Project 4)
- `tools.py` — standalone tool functions: `web_search`, `calculator`, `wikipedia_lookup`, `rag_retrieve`
- `agent.py` — LangGraph loop: `agent` node (LLM picks tool or answers) <-> `tool` node (executes it)
- `requirements.txt`

## Setup
```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
```
Get a free key from [console.groq.com](https://console.groq.com). The
default model is `llama-3.3-70b-versatile` (set in `agent.py`) — check
Groq's docs for other models that support tool calling if you want to swap it.

## Run
```bash
python agent.py
```
This runs a built-in multi-hop test question. Edit the `if __name__` block
or call `run("your question")` from another script.

## How the loop works
1. `agent_node` sends the conversation + tool definitions to Groq.
2. If the model returns one or more `tool_calls`, `tool_node` runs each
   one and appends the result as a separate `{"role": "tool", ...}`
   message (matched back via `tool_call_id`), then loops back to `agent_node`.
3. If the model responds with plain text and no `tool_calls`, that's
   treated as the final answer and the graph ends — this can happen on
   the very first pass if the question doesn't need any tool at all.
4. `MAX_ITERATIONS` (default 5) stops runaway loops — tune this per use case.

## Next steps to make this yours
- **Test each tool alone first**: `python -c "from tools import *; print(web_search(WebSearchInput(query='test')))"`
- **Wire in your real RAG store**: `rag_retrieve` expects a ChromaDB collection
  named `documents` at `./chroma_db` — point it at your PDF Q&A bot's existing DB.
- **Swap `web_search`** for a real search API (Tavily/Serper/Bing) — the DuckDuckGo
  instant-answer API used here is free but weak on many queries.
- **Add tracing**: print `state["messages"]` after each node to watch the
  agent's reasoning/tool-call trail — invaluable for debugging loops.
- **Try a deliberately multi-hop question** to confirm looping actually
  happens, not just single tool calls.
