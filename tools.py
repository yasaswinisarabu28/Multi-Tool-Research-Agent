"""
Standalone tools. Each one is a plain Python function that takes simple
args and returns a string. No LLM logic lives in here — keep these
dumb and testable on their own before wiring them into the agent.
"""

import ast
import operator

import requests

from schemas import CalculatorInput, RagRetrieveInput, WebSearchInput, WikipediaInput


# ---------------------------------------------------------------------------
# 1. Web search — using DuckDuckGo's free instant-answer API (no key needed).
#    Swap this for a real search API (Tavily, Serper, Bing) for better recall.
# ---------------------------------------------------------------------------
def web_search(args: WebSearchInput) -> str:
    try:
        from tavily import TavilyClient
        tavily_client = TavilyClient()  # reads TAVILY_API_KEY from environment
        results = tavily_client.search(query=args.query, max_results=3)
        snippets = [r["content"] for r in results.get("results", [])]
        return " | ".join(snippets) if snippets else f"No results found for '{args.query}'."
    except Exception as e:
        return f"web_search error: {e}"

# ---------------------------------------------------------------------------
# 2. Calculator — safe expression evaluation (no raw eval()).
# ---------------------------------------------------------------------------
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


def calculator(args: CalculatorInput) -> str:
    try:
        tree = ast.parse(args.expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"calculator error: {e}"


# ---------------------------------------------------------------------------
# 3. Wikipedia lookup — REST summary endpoint, no API key needed.
# ---------------------------------------------------------------------------
def wikipedia_lookup(args: WikipediaInput) -> str:
    try:
        title = args.topic.strip().replace(" ", "_")
        resp = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
            headers={"User-Agent": "MultiToolResearchAgent/1.0 (student project)"},
            timeout=10,
        )
        
        if resp.status_code != 200:
            return f"No Wikipedia page found for '{args.topic}'."
        data = resp.json()
        return data.get("extract", f"No summary available for '{args.topic}'.")
    except Exception as e:
        return f"wikipedia_lookup error: {e}"


# ---------------------------------------------------------------------------
# 4. RAG retrieve — plug in your existing ChromaDB collection from the
#    PDF Q&A bot. Stubbed here so this module runs standalone.
# ---------------------------------------------------------------------------
def rag_retrieve(args: RagRetrieveInput) -> str:
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer

        client = chromadb.PersistentClient(path="./chroma_db")
        collection = client.get_collection("documents")  # match your PDF bot's name
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_embedding = model.encode([args.query]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=args.top_k)
        chunks = results.get("documents", [[]])[0]
        return "\n---\n".join(chunks) if chunks else "No relevant chunks found."
    except Exception as e:
        return f"rag_retrieve error (is your ChromaDB set up at ./chroma_db?): {e}"


# Registry the agent will use to map tool name -> (function, schema)
TOOL_REGISTRY = {
    "web_search": (web_search, WebSearchInput),
    "calculator": (calculator, CalculatorInput),
    "wikipedia_lookup": (wikipedia_lookup, WikipediaInput),
    "rag_retrieve": (rag_retrieve, RagRetrieveInput),
}
