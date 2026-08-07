import json
import os
from tavily import TavilyClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain_core.documents import Document
from .state import AgentState
from .utils import _llm, logger

# Initialize Clients
tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

_vector_cache = None

def get_vector_cache():
    global _vector_cache
    if _vector_cache is not None:
        return _vector_cache
    
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key or api_key == "your_pinecone_api_key_here":
        logger.warning("⚠️ [Pinecone] PINECONE_API_KEY not configured. Vector cache is disabled.")
        return None

    try:
        pc = PineconeClient(api_key=api_key)
        index_name = os.environ.get("PINECONE_INDEX_NAME", "search-cache")
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        
        if index_name not in existing_indexes:
            logger.info(f"🌲 [Pinecone] Index '{index_name}' not found. Creating serverless index (dim=768, metric='cosine')...")
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logger.info(f"✅ [Pinecone] Index '{index_name}' created successfully.")
            
        index = pc.Index(index_name)
        _vector_cache = PineconeVectorStore(
            index=index,
            embedding=embeddings
        )
        return _vector_cache
    except Exception as e:
        logger.warning(f"⚠️ [Pinecone] Vector cache initialization failed: {e}")
        return None

# ── Analyzer ──────────────────────────────────
def analyzer_node(state: AgentState):
    raw = state["query"]
    logger.info(f"===> [NODE START: Analyzer] Processing query: '{raw}'")
    
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
    except Exception as e:
        logger.warning(f"⚠️ [Analyzer] Failed to parse LLM JSON output ({e}). Falling back to defaults.")
        parsed = {"rewritten_query": raw, "queries": [raw], "plan": "Search directly.", "needs_search": True}
    
    rewritten = parsed.get('rewritten_query', raw)
    sub_queries = parsed.get('queries', [raw])
    needs_search = parsed.get('needs_search', True)
    
    logger.info(f"✏️  [Analyzer] Rewritten Query: '{rewritten}'")
    logger.info(f"📋 [Analyzer] Generated {len(sub_queries)} sub-queries: {sub_queries}")
    logger.info(f"🔍 [Analyzer] Live search required: {needs_search}")
    logger.info("<=== [NODE END: Analyzer]\n")
    
    return {**parsed, "reflection_count": 0, "agent": "Analyzer Agent"}

# ── Cache Check ───────────────────────────────
def cache_check_node(state: AgentState):
    query = state["query"]
    logger.info(f"===> [NODE START: Cache Check] Checking Pinecone cache for: '{query}'")
    
    cache = get_vector_cache()
    if cache is not None:
        try:
            docs = cache.similarity_search_with_score(query, k=1)
            if docs:
                doc, score = docs[0]
                if score >= 0.80:
                    logger.info(f"⚡ [Cache Check] HIT! Similarity score: {score:.4f} (>= 0.80) — Skipping live search")
                    summary_val = doc.metadata.get("summary", doc.page_content)
                    citations_val = json.loads(doc.metadata.get("citations", "[]"))
                    logger.info(f"📚 [Cache Check] Retrieved cached summary ({len(summary_val):,} chars, {len(citations_val)} citations)")
                    logger.info("<=== [NODE END: Cache Check]\n")
                    return {"summarized_results": summary_val, "needs_search": False, "citations": citations_val}
                else:
                    logger.info(f"🔍 [Cache Check] Similarity score {score:.4f} below threshold (0.80)")
        except Exception as e:
            logger.warning(f"⚠️ [Cache Check] Error querying Pinecone: {e}")
    
    logger.info("🔍 [Cache Check] MISS — Proceeding to live web search via Tavily")
    logger.info("<=== [NODE END: Cache Check]\n")
    return {"needs_search": True, "agent": "Cache Agent"}


# ── Search ────────────────────────────────────
def search_node(state: AgentState):
    queries = state.get("queries") or [state["rewritten_query"]]
    logger.info(f"===> [NODE START: Search] Executing live search for {len(queries)} sub-queries")
    
    all_results = []
    citations = []
    
    for idx, q in enumerate(queries, 1):
        logger.info(f"🌐 [Search] Sub-query #{idx}/{len(queries)}: '{q}'")
        try:
            response = tavily.search(query=q, search_depth="advanced", max_results=2)
            r_list = response.get("results", [])
            logger.info(f"  └─ Fetched {len(r_list)} web results for sub-query #{idx}")
            all_results.extend([r.get("content", "") for r in r_list])
            for r in r_list:
                citations.append({
                    "title": r.get("title", ""),
                    "source_url": r.get("url", ""),
                    "claim": ""
                })
        except Exception as e:
            logger.error(f"  ⚠️ [Search] Tavily search error for '{q}': {e}")

    logger.info(f"✅ [Search] Completed. Total sources collected: {len(citations)}")
    logger.info("<=== [NODE END: Search]\n")
    return {"search_results": all_results, "citations": citations, "agent": "Search Agent"}

# ── Summarizer ────────────────────────────────
def summarizer_node(state: AgentState):
    logger.info("===> [NODE START: Summarizer] Processing raw search results")
    raw_data = str(state.get("search_results", []))
    
    if not raw_data.strip() or raw_data == "[]":
        logger.warning("⚠️ [Summarizer] No search results found to summarize.")
        logger.info("<=== [NODE END: Summarizer]\n")
        return {"summarized_results": "No search results available."}

    logger.info(f"📝 [Summarizer] Compressing {len(raw_data):,} characters of raw data via LLM...")
    summary = _llm(
        "You are a precise data extractor.", 
        f"Compress these search results into a dense summary (max 1500 words). Preserve all numbers and dates.\n\nData: {raw_data[:10000]}",
        agent="Summarizer Agent"
    )
    
    citations = state.get("citations", [])
    cache = get_vector_cache()
    if cache is not None:
        try:
            cache.add_documents([Document(page_content=state["query"], metadata={"summary": summary, "citations": json.dumps(citations)})])
            logger.info("💾 [Summarizer] Saved compressed summary to Pinecone vector cache")
        except Exception as e:
            logger.warning(f"⚠️ [Summarizer] Cache save failed: {e}")

        
    logger.info(f"📝 [Summarizer] Compression complete: {len(raw_data):,} chars -> {len(summary):,} chars")
    logger.info("<=== [NODE END: Summarizer]\n")
    return {"summarized_results": summary, "agent": "Summarizer Agent"}

# ── Research Node ─────────────────────────────
def research_node(state: AgentState):
    logger.info("===> [NODE START: Research] Synthesizing initial report draft")
    context = state["summarized_results"]
    query = state["rewritten_query"]
    plan = state.get("plan", "")
    
    logger.info(f"📖 [Research] Generating report with {len(context):,} chars of context...")
    response = _llm(
        "You are an expert research analyst.",
        f"Research Plan:\n{plan}\n\nData:\n{context}\n\nQuery: {query}\n\nWrite a structured answer with clear headings. End with 'CONFIDENCE: 0.X'.",
        agent="Research Agent"
    )   
    logger.info(f"📄 [Research] Draft response ready ({len(response):,} chars)")
    logger.info("<=== [NODE END: Research]\n")
    return {"response": response, "agent": "Research Agent"}

# ── Critique Node ─────────────────────────────
def critique_node(state: AgentState):
    count = state.get("reflection_count", 0) + 1
    logger.info(f"===> [NODE START: Critique] Reviewing report quality (Iteration #{count})")
    
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
        suggestions = parsed.get("suggestions", "No suggestions provided.")
    except Exception as e:
        logger.warning(f"⚠️ [Critique] Failed to parse critique JSON ({e}). Defaulting to PASS.")
        verdict, score, suggestions = "PASS", 0.8, "Default pass."

    logger.info(f"🔍 [Critique] Verdict: {verdict} | Score: {score:.2f} | Iteration #{count}")
    logger.info(f"💡 [Critique] Feedback suggestions: {suggestions}")
    logger.info("<=== [NODE END: Critique]\n")
    return {"critique": critique, "reflection_count": count, "confidence_score": score, "agent": "Critique Agent"}

# ── Reviser Node ──────────────────────────────
def reviser_node(state: AgentState):
    logger.info("===> [NODE START: Reviser] Refining draft based on critique feedback")
    original = state["response"]
    feedback = state["critique"]
    
    revised = _llm(
        "You are a senior editor.", 
        f"Improve this report based on feedback: {feedback}\n\nOriginal: {original}",
        agent="Reviser Agent"
    )
    logger.info(f"♻️  [Reviser] Response revised ({len(original):,} chars -> {len(revised):,} chars)")
    logger.info("<=== [NODE END: Reviser]\n")
    return {"response": revised, "agent": "Reviser Agent"}

# ── Outputter Node ────────────────────────────
def output_formatter_node(state: AgentState):
    logger.info("===> [NODE START: Output Formatter] Generating final markdown document")
    resp = state["response"]
    existing_citations = state.get("citations", [])
    
    final = _llm(
        "You are a professional technical writer and document designer.", 
        f"Transform the following research data into a beautifully formatted Markdown report. "
        f"Use clear headings, bullet points, and bold text for emphasis. "
        f"IMPORTANT: The 'final_response' field MUST be a single string containing the full Markdown text. "
        f"Format the final output as a JSON object with keys 'final_response' and 'citations'. "
        f"For the 'citations' field, review the following sources and select the TOP 3 most relevant ones that best support the claims in the report: {existing_citations}\n\nInput: {resp}",
        agent="Output Agent"
    )
    try:
        parsed = json.loads(final.strip().strip("```json").strip("```"))
        if not parsed.get("citations") and existing_citations:
            parsed["citations"] = existing_citations
    except Exception as e:
        logger.warning(f"⚠️ [Output Formatter] Failed to parse JSON formatted report ({e}).")
        parsed = {"final_response": resp, "citations": existing_citations}
    
    logger.info(f"✅ [Output Formatter] Final document completed ({len(parsed.get('final_response', '')):,} chars, {len(parsed.get('citations', []))} citations)")
    logger.info("<=== [NODE END: Output Formatter]\n")
    return {**parsed, "agent": "Output Agent"}

# ── Routers ───────────────────────────────────
def route_after_planner(state: AgentState):
    target = "research" if not state.get("needs_search") else "cache_check"
    logger.info(f"🔀 [ROUTER: after_planner] Decision: needs_search={state.get('needs_search')} ---> Routing to '{target}'")
    return target

def route_after_cache(state: AgentState):
    target = "search" if state.get("needs_search") else "research"
    logger.info(f"🔀 [ROUTER: after_cache] Decision: needs_search={state.get('needs_search')} ---> Routing to '{target}'")
    return target

def route_after_critique(state: AgentState):
    try:
        crit = json.loads(state.get("critique", "{}"))
        verdict = crit.get("verdict", "PASS")
    except:
        verdict = "PASS"
    
    count = state.get("reflection_count", 0)
    if verdict == "PASS" or count >= 2: 
        target = "outputter"
        logger.info(f"🔀 [ROUTER: after_critique] Decision: verdict='{verdict}', reflection_count={count} ---> Routing to '{target}'")
    else:
        target = "reviser"
        logger.info(f"🔀 [ROUTER: after_critique] Decision: verdict='{verdict}', reflection_count={count} ---> Routing to '{target}'")
    return target
