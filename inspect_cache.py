import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone as PineconeClient
from langchain_pinecone import Pinecone as PineconeVectorStore

load_dotenv()

def inspect():
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME", "search-cache")

    if not api_key or api_key == "your_pinecone_api_key_here":
        print("❌ PINECONE_API_KEY is missing or set to placeholder in .env. Please configure your Pinecone API key.")
        return

    print(f"\n🌲 PINECONE CACHE INSPECTION: Index '{index_name}'")
    try:
        pc = PineconeClient(api_key=api_key)
        indexes = [idx.name for idx in pc.list_indexes()]
        
        if index_name not in indexes:
            print(f"❌ Index '{index_name}' does not exist in your Pinecone project yet.")
            return

        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        total_vectors = stats.get("total_vector_count", 0)
        print(f"Total vector entries found: {total_vectors}")
        print("-" * 50)

        if total_vectors > 0:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            vector_cache = PineconeVectorStore(
                index=index,
                embedding=embeddings
            )
            # Perform a broad search to inspect cached documents
            sample_docs = vector_cache.similarity_search("", k=min(20, total_vectors))
            for i, doc in enumerate(sample_docs):
                query = doc.page_content
                summary = doc.metadata.get("summary", "No summary available")
                content_preview = summary[:150].replace("\n", " ") + "..."
                print(f"[{i+1}] 🎯 CACHED QUERY: {query}")
                print(f"    📄 SUMMARY PREVIEW: {content_preview}")
                print("-" * 50)
    except Exception as e:
        print(f"⚠️ Error inspecting Pinecone index: {e}")

if __name__ == "__main__":
    inspect()

