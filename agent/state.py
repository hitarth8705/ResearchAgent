from typing import TypedDict, List, Optional

class AgentState(TypedDict):
    """The state of the research agent."""
    agent: str
    query: str
    rewritten_query: str
    queries: List[str]
    plan: str
    search_results: List[dict]
    summarized_results: str
    response: str
    reflection_count: int
    critique: str
    confidence_score: float
    needs_search: bool
    final_response: str
    citations: List[dict]
