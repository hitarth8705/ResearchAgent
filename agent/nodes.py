import json
import os
from tavily import TavilyClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from .state import AgentState
from .utils import _llm

# Initialize Clients
tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_cache = Chroma(
    collection_name="search_cache",
    embedding_function=embeddings,
    persist_directory="./agent_cache"
)

# ── Analyzer ──────────────────────────────────
def analyzer_node(state: AgentState):
    raw = state["query"]
    result = _llm(
        system="You are an expert research strategist. Accuracy and data-driven analysis are your top priorities.",
        user=f"""User Query: "{raw}"
Perform the following tasks:
1. **Rewrite**: Optimize for clarity and technical depth.
2. **Decompose**: Create exactly 3 focused sub-queries for evidence gathering.
3. **Plan**: Create a step-by-step research strategy (Format as a Markdown numbered list with newlines).
4. **Search Decision**: Decide if live search is mandatory.

Respond ONLY in JSON with 'rewritten_query', 'queries', 'plan', and 'needs_search' (bool).""",
        agent="Analyzer Agent"
    )
    try:
        parsed = json.loads(result.strip().strip("```json").strip("```"))
    except:
        parsed = {"rewritten_query": raw, "queries": [raw], "plan": "Search directly.", "needs_search": True}
    
    print(f"\n✏️  Rewritten Query: {parsed.get('rewritten_query', raw)}")
    print(f"📋 Plan ready | Search needed: {parsed.get('needs_search', True)}")
    return {**parsed, "reflection_count": 0, "agent": "Analyzer Agent"}

# ── Cache Check ───────────────────────────────
def cache_check_node(state: AgentState):
    query = state["query"]
    docs = vector_cache.similarity_search_with_score(query, k=1)
    if docs and docs[0][1] < 0.5:
        print("⚡ Cache HIT — skipping Tavily search")
        return {"summarized_results": docs[0][0].metadata.get("summary", docs[0][0].page_content), "needs_search": False, "citations": json.loads(docs[0][0].metadata.get("citations", "[]"))}
    print("🔍 Cache MISS — proceeding to Tavily")
    return {"needs_search": True, "agent": "Cache Agent"}

# ── Search ────────────────────────────────────
def search_node(state: AgentState):
    queries = state.get("queries") or [state["rewritten_query"]]
    all_results = []
    citations = []
    
    for q in queries:
        try:
            response = tavily.search(query=q, search_depth="advanced", max_results=2)
            r_list = response.get("results", [])
            all_results.extend([r.get("content", "") for r in r_list])
            for r in r_list:
                citations.append({
                    "title": r.get("title", ""),
                    "source_url": r.get("url", ""),
                    "claim": ""
                })
        except Exception as e:
            print(f"  ⚠️ Tavily error for '{q}': {e}")

    print(f"🌐 Search complete | {len(citations)} sources collected")
    return {"search_results": all_results, "citations": citations, "agent": "Search Agent"}

# ── Summarizer ────────────────────────────────
def summarizer_node(state: AgentState):
    raw_data = str(state.get("search_results", []))
    if not raw_data.strip() or raw_data == "[]":
        return {"summarized_results": "No search results available."}

    summary = _llm(
        "You are a precise data extractor.", 
        f"Compress these search results into a dense summary (max 1500 words). Preserve all numbers and dates.\n\nData: {raw_data[:10000]}",
        agent="Summarizer Agent"
    )
    
    citations = state.get("citations", [])
    vector_cache.add_documents([Document(page_content=state["query"], metadata={"summary": summary, "citations": json.dumps(citations)})])
    print(f"📝 Summarized: {len(raw_data):,} → {len(summary):,} chars")
    return {"summarized_results": summary, "agent": "Summarizer Agent"}

# ── Research Node ─────────────────────────────
def research_node(state: AgentState):
    context = state["summarized_results"]
    query = state["rewritten_query"]
    plan = state.get("plan", "")
    
    response = _llm(
        "You are an expert research analyst.",
        f"Research Plan:\n{plan}\n\nData:\n{context}\n\nQuery: {query}\n\nWrite a structured answer with clear headings. End with 'CONFIDENCE: 0.X'.",
        agent="Research Agent"
    )   
    print("📄 Draft response ready")
    return {"response": response, "agent": "Research Agent"}

# ── Critique Node ─────────────────────────────
def critique_node(state: AgentState):
    resp = state["response"]
    critique = _llm(
        "You are a rigorous quality reviewer.", 
        f"Review this report for accuracy and depth. Respond in JSON with 'verdict' (PASS/REVISE), 'score' (0-1), and 'suggestions'.\n\nReport: {resp}",
        agent="Critique Agent"
    )
    
    try:
        parsed = json.loads(critique.strip().strip("```json").strip("```"))
        verdict = parsed.get("verdict", "PASS")
        score = float(parsed.get("score", 0.8))
    except:
        verdict, score = "PASS", 0.8

    count = state.get("reflection_count", 0) + 1
    print(f"🔍 Critique: {verdict} | Score: {score:.2f} | Loop #{count}")
    return {"critique": critique, "reflection_count": count, "confidence_score": score, "agent": "Critique Agent"}

# ── Reviser Node ──────────────────────────────
def reviser_node(state: AgentState):
    original = state["response"]
    feedback = state["critique"]
    revised = _llm(
        "You are a senior editor.", 
        f"Improve this report based on feedback: {feedback}\n\nOriginal: {original}",
        agent="Reviser Agent"
    )
    print("♻️  Response revised")
    return {"response": revised, "agent": "Reviser Agent"}

# ── Outputter Node ────────────────────────────
def output_formatter_node(state: AgentState):
    resp = state["response"]
    existing_citations = state.get("citations", [])
    
    final = _llm(
        "You are a professional technical writer and document designer.", 
        f"Transform the following research data into a beautifully formatted Markdown report. "
        f"Use clear headings, bullet points, and bold text for emphasis. "
        f"IMPORTANT: The 'final_response' field MUST be a single string containing the full Markdown text. "
        f"Format the final output as a JSON object with keys 'final_response' and 'citations'. "
        f"For the 'citations' field, review the following  sources and select the TOP 3  most relevant ones that best support the claims in the report: {existing_citations}\n\nInput: {resp}",
        agent="Output Agent"
    )
    try:
        parsed = json.loads(final.strip().strip("```json").strip("```"))
        # If the LLM didn't return any citations but we have them, keep the originals
        if not parsed.get("citations") and existing_citations:
            parsed["citations"] = existing_citations
    except:
        parsed = {"final_response": resp, "citations": existing_citations}
    
    print("\n✅ Final report ready")
    return {**parsed, "agent": "Output Agent"}

# ── Routers ───────────────────────────────────
def route_after_planner(state: AgentState):
    return "research" if not state.get("needs_search") else "cache_check"

def route_after_cache(state: AgentState):
    return "search" if state.get("needs_search") else "research"

def route_after_critique(state: AgentState):
    try:
        crit = json.loads(state.get("critique", "{}"))
        verdict = crit.get("verdict", "PASS")
    except:
        verdict = "PASS"
    
    if verdict == "PASS" or state.get("reflection_count", 0) >= 2: 
        return "outputter"
    return "reviser"
