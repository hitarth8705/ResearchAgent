import os
import sys
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from .state import AgentState
from .nodes import (
    analyzer_node, cache_check_node, search_node, summarizer_node,
    research_node, critique_node, reviser_node, output_formatter_node,
    route_after_planner, route_after_cache, route_after_critique
)

# In-memory checkpointer for conversation memory
memory = MemorySaver()

builder = StateGraph(AgentState)

# ── Nodes ─────────────────────────────────────
builder.add_node("analyzer",    analyzer_node)
builder.add_node("cache_check", cache_check_node)
builder.add_node("search",      search_node)
builder.add_node("summarizer",  summarizer_node)
builder.add_node("research",    research_node)
builder.add_node("critique",    critique_node)
builder.add_node("reviser",     reviser_node)
builder.add_node("outputter",   output_formatter_node)

# ── Edges ─────────────────────────────────────
builder.set_entry_point("analyzer")

builder.add_edge("search",      "summarizer")
builder.add_edge("summarizer",  "research")
builder.add_edge("research",    "critique")
builder.add_edge("reviser",     "research")
builder.add_edge("outputter",   END)

# ── Conditional Routing ────────────────────────
builder.add_conditional_edges(
    "analyzer",
    route_after_planner,
    {"cache_check": "cache_check", "research": "research"}
)

builder.add_conditional_edges(
    "cache_check",
    route_after_cache,
    {"search": "search", "research": "research"}
)

builder.add_conditional_edges(
    "critique",
    route_after_critique,
    {"reviser": "reviser", "outputter": "outputter"}
)

# ── Compile ───────────────────────────────────
# Check if running in LangGraph Studio/Dev server
is_server = any(os.environ.get(k) for k in ["LANGGRAPH_API_VERSION", "LANGGRAPH_CLOUD", "LANGGRAPH_DEV"])

if is_server or "langgraph_api" in sys.modules:
    graph = builder.compile()
else:
    graph = builder.compile(checkpointer=memory)
