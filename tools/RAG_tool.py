import json
import os
from dotenv import load_dotenv

from chromadb import PersistentClient
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from utils.BaseModels import RetrievedKnowledge

load_dotenv(override=True)

api_key = os.getenv("GOOGLE_API_KEY")


@tool
def lookup_university_smart(query: str):
    """
    Performs a semantic search in the university knowledge base.
    Uses an internal LLM to analyze retrieved documents and extract the Official URL and Name.
    Use this to find the URL for a university.
    """
    if api_key:
        print(f"🔴 DEBUG: Key currently in use starts with: {api_key[:30]}...")
        print(f"🔴 DEBUG: Key length: {len(api_key)}")
    else:
        print("🔴 DEBUG: No GOOGLE_API_KEY found in environment!")


    print(f"🔍 RAG Tool: Searching knowledge base for '{query}'...")

    try:
        # --- A. RETRIEVAL PHASE ---
        persist_dir = "./chroma_db"
        collection_name = "university_db"

        # Initialize Client & Embeddings
        chroma_client = PersistentClient(path=persist_dir)
        collection = chroma_client.get_or_create_collection(name=collection_name)

        # Embed the query
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=api_key)
        query_vector = embeddings.embed_query(query)

        # Query Chroma (Fetch top 4 chunks for broader context)
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=4
        )

        # Flatten documents
        docs = results.get("documents", [[]])[0]
        if not docs:
            return json.dumps({"found": False, "reason": "No documents in database."})

        raw_context = "\n\n---\n\n".join(docs)

        # --- B. READER/REASONING PHASE (The Internal LLM) ---
        # We spin up a lightweight LLM just to process this specific data
        reader_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            api_key=api_key
        )

        # Bind the schema to force structured analysis
        structured_reader = reader_llm.with_structured_output(RetrievedKnowledge)

        # The Prompt specifically for the RAG task
        rag_prompt = f"""
        You are a Fact Extraction Agent. 
        Analyze the following retrieved context chunks regarding a university.

        User Query: {query}

        Retrieved Context:
        {raw_context}

        Task:
        1. Identify the Official University Name.
        2. Extract the main Official Website URL (look for http/https, or patterns like 'visit ucp.edu').
        3. Summarize key info.

        If the context is irrelevant to the query, set 'found' to False.
        """

        # Invoke the internal chain
        extraction = structured_reader.invoke(rag_prompt)

        # --- C. RETURN TO CHAT NODE ---
        # We return the JSON string so the Main Chat LLM can read it as a tool output
        return extraction.json()

    except Exception as e:
        return json.dumps({"found": False, "error": str(e)})
