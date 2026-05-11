import os
import time
import json
import uuid
import streamlit as st
from agent import graph, token_tracker
from agent.state import AgentState
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ─────────────────────────────────────────────
# 1. FORMATTING UTILS
# ─────────────────────────────────────────────
def format_content(content):
    """Ensure content is rendered as Markdown, converting dicts if necessary."""
    if isinstance(content, dict):
        md = ""
        for key, value in content.items():
            if key.lower() == "title":
                md += f"# {value}\n\n"
            elif key.lower() == "sections" and isinstance(value, list):
                for section in value:
                    if isinstance(section, dict):
                        md += f"## {section.get('heading', 'Section')}\n\n"
                        md += f"{section.get('content', '')}\n\n"
            else:
                md += f"### {key.capitalize()}\n\n{value}\n\n"
        return md
    return str(content)

# ─────────────────────────────────────────────
# 2. RUNNER UTILS
# ─────────────────────────────────────────────
def run_agent(query: str, thread_id: str = "default", verbose: bool = True) -> dict:
    """Run the research agent with node tracking."""
    token_tracker.reset()
    config = {"configurable": {"thread_id": thread_id}}

    # Initial state initialization
    initial_state = {
        "query": query,
        "queries": [],
        "agent": "Analyzer",
        "reflection_count": 0,
        "needs_search": True
    }

    print(f"\n🚀 RESEARCH AGENT | Query: {query[:80]}...")
    
    # Stream the graph execution
    for step in graph.stream(initial_state, config=config, stream_mode="updates"):
        for node_name, _ in step.items():
            if verbose:
                print(f"  ▶ {node_name.upper()} node complete")

    return graph.get_state(config).values

def pretty_print(state: dict):
    """Print the final research report to console."""
    print("\n" + "="*60)
    print("🎓 FINAL RESEARCH REPORT")
    print("="*60)
    
    # Check for either final_response or full_response
    content = state.get("final_response") or state.get("full_response") or "No response generated."
    print(f"\n{content}")
    
    print("\n" + "="*60)
    print("📎 CITATIONS")
    citations = state.get("citations", [])
    for i, c in enumerate(citations, 1):
        print(f"  {i}. {c.get('title', 'Source')}\n     URL: {c.get('source_url', 'N/A')}")

# ─────────────────────────────────────────────
# 2. STREAMLIT UI
# ─────────────────────────────────────────────
def launch_streamlit_ui():
    st.set_page_config(page_title="Advanced Research Agent", page_icon="🔬", layout="wide")
    
    # Premium Styling
    st.markdown("""
        <style>
        .stApp { background-color: #0e1117; color: #e0e0e0; }
        .main-header { font-size: 2.2rem; font-weight: 700; color: #00d4ff; margin-bottom: 20px; text-align: center; }
        .metric-card { background-color: #1e2130; padding: 15px; border-radius: 10px; border-left: 5px solid #00d4ff; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">🔬 Advanced AI Research Agent</div>', unsafe_allow_html=True)

    # Initialize Session State
    if "results" not in st.session_state:
        st.session_state.results = None
    if "metrics" not in st.session_state:
        st.session_state.metrics = {"calls": 0, "cost": 0.0, "summary": ""}

    # Sidebar Metrics
    with st.sidebar:
        st.header("📊 Performance Metrics")
        st.info("Powered by LangGraph & Gemini 2.5")
        st.metric("Total LLM Calls", st.session_state.metrics["calls"])
        st.metric("Est. Cost (USD)", f"${st.session_state.metrics['cost']:.5f}")
        st.divider()
        if st.session_state.metrics["summary"]:
            st.text(st.session_state.metrics["summary"])
        
        if st.button("🗑️ Clear History"):
            st.session_state.results = None
            st.session_state.metrics = {"calls": 0, "cost": 0.0, "summary": ""}
            st.rerun()

    # Input Section
    query = st.chat_input("Enter your research topic...")
    
    if query:
        with st.status("🏗️ Agent working...", expanded=True) as status:
            try:
                final_state = run_agent(query, thread_id=str(uuid.uuid4()), verbose=True)
                
                # Store results
                st.session_state.results = final_state
                st.session_state.metrics = {
                    "calls": token_tracker.total_calls,
                    "cost": token_tracker.estimated_cost_usd,
                    "summary": token_tracker.summary()
                }
                status.update(label="✅ Success!", state="complete", expanded=False)
                st.rerun()
            except Exception as e:
                status.update(label="❌ Error occurred", state="error", expanded=True)
                st.error(f"Unexpected error: {str(e)}")

    # Display results
    if st.session_state.results:
        output = st.session_state.results
        
        tab1, tab2, tab3 = st.tabs(["📄 Detailed Report", "📎 Cited Sources", "📋 Strategy"])
        
        with tab1: 
            # Check for final_response (modular name) or full_response (old name)
            content = output.get("final_response") or output.get("full_response", "")
            st.markdown(format_content(content))
            
        with tab2:
            citations = output.get("citations", [])
            if citations:
                for i, c in enumerate(citations, 1):
                    st.markdown(f"**{i}. {c.get('title','Source')}**")
                    st.markdown(f"[Source Link]({c.get('source_url','')})")
                    st.divider()
            else: 
                st.write("No citations available.")
                
        with tab3: 
            st.markdown(output.get("plan", "No plan available."))

        st.download_button(
            "📥 Save as JSON", 
            json.dumps(output, indent=2), 
            "research_report.json", 
            "application/json"
        )

if __name__ == "__main__":
    import sys
    # Check if run through streamlit
    if "streamlit" in sys.modules or (len(sys.argv) > 1 and sys.argv[1] == "run"):
        launch_streamlit_ui()
    else:
        print("\n💡 Run 'streamlit run main.py' for UI\n")
        res = run_agent("Future of AI Agent Frameworks")
        pretty_print(res)
