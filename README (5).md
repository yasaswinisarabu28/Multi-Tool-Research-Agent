# Multi-Tool Research Agent — Scaffold

## Files
- `schemas.py` — Pydantic input schemas per tool (reused as MCP schemas in Project 4)
- `tools.py` — standalone tool functions: `web_search`, `calculator`, `wikipedia_lookup`, `rag_retrieve`
- `agent.py` — LangGraph loop: `agent` node (LLM picks tool or answers) <-> `tool` node (executes it)
- `requirements.txt`

## Setup
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
```

## Run
```bash
python agent.py
```
This runs a built-in multi-hop test question. Edit the `if __name__` block
or call `run("your question")` from another script.

## How the loop works
1. `agent_node` sends the conversation + tool definitions to the model.
2. If the model calls a tool, `tool_node` runs it and appends the result
   as a `tool_result` message, then loops back to `agent_node`.
3. If the model responds with plain text instead of a tool call, that's
   treated as the final answer and the graph ends.
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
