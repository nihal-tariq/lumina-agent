import os
import json

from chromadb import PersistentClient
from google import genai

from state import State

GEMINI_API_KEY= os.environ.get("GEMINI_API_KEY")

client = genai.Client()


def rag_agent_node(state: State) -> State:
    """
    Agent node that:
      - retrieves (RAG) context from Chroma (if available)
      - builds a prompt combining last user message + retrieved context
      - asks Gemini to return structured JSON with UniversityName, URL, URL_info
    This node does NOT scrape; it only returns the structured fields.
    """
    msgs = state.get("messages", [])
    last_user = None
    # find last user message
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("role") == "user":
            last_user = m.get("content")
            break
    if last_user is None:
        # nothing to act on
        return {}

    # --- RAG retrieval (Chroma) ---
    # This step is optional — it will attempt to retrieve context from ./chroma_db
    try:
        persist_dir = "./chroma_db"
        collection_name = "university_db"
        chroma_client = PersistentClient(path=persist_dir)
        collection = chroma_client.get_or_create_collection(name=collection_name)
        # For RAG we embed last_user via Gemini embeddings and query Chroma
        embed_result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=last_user
        )
        query_vector = embed_result.embeddings[0].values
        chroma_results = collection.query(query_embeddings=[query_vector], n_results=3)
        docs = chroma_results.get("documents", [[]])[0]
        chroma_context = "\n\n".join(docs) if docs else ""
    except Exception as e:
        chroma_context = ""  # best-effort retrieval; agent still proceeds

    # Build a clear instruction for structured output
    prompt = f"""
You are an assistant that MUST respond with a single valid JSON object (no extra text)
with exact fields:
{{
  "UniversityName": string,
  "URL": [list of strings],
  "URL_info": [list of strings]
}}

User Request: {last_user}

Retrieved context (for reference; may be empty):
{chroma_context}

Rules:
- If the user provided URLs in their message, extract them into the "URL" array.
- If no URL is found, return "URL": [] and add a message in "URL_info" asking for the URL.
- Keep JSON valid and parsable.
- Do NOT include explanatory text outside of the JSON.
"""

    msg = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )

    response_text = msg.text if hasattr(msg, "text") else str(msg)
    # try to parse JSON — Gemini sometimes returns extra whitespace, try to locate the JSON
    parsed = None
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        # attempt to extract the first JSON object in the text
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            maybe = response_text[start:end+1]
            try:
                parsed = json.loads(maybe)
            except json.JSONDecodeError:
                parsed = None

    if not parsed:
        # fallback: ask user for URL (structured minimal)
        parsed = {
            "UniversityName": "",
            "URL": [],
            "URL_info": ["LLM failed to produce valid JSON. Please provide the website URL(s)."]
        }

    # Ensure URL is list
    urls = parsed.get("URL", [])
    if isinstance(urls, str):
        urls = [urls]

    return {
        "UniversityName": parsed.get("UniversityName", ""),
        "URL": urls,
        "URL_info": parsed.get("URL_info", []),
        "chroma_context": chroma_context
    }