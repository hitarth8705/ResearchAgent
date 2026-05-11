import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def inspect():
    # 1. Initialize the same embeddings and cache folder
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    if not os.path.exists("./agent_cache"):
        print("❌ No cache folder found yet. Run a search first!")
        return

    vector_cache = Chroma(
        collection_name="search_cache",
        embedding_function=embeddings,
        persist_directory="./agent_cache"
    )

    # 2. Get all data
    # Note: .get() returns all IDs, documents, and metadatas
    data = vector_cache.get()
    
    ids = data.get("ids", [])
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    print(f"\n📂 CHROMA CACHE INSPECTION")
    print(f"Total entries found: {len(ids)}")
    print("-" * 50)

    for i in range(len(ids)):
        query = metadatas[i].get("query", "Unknown Query")
        content_preview = documents[i][:150].replace("\n", " ") + "..."
        
        print(f"[{i+1}] 🎯 QUERY: {query}")
        print(f"    📄 CONTENT: {content_preview}")
        print("-" * 50)

if __name__ == "__main__":
    inspect()
