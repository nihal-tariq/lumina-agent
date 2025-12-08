import os
import re
import uuid
import pandas as pd
import chromadb
from typing import List
from google import genai
from chromadb import PersistentClient


def get_or_create_knowledge_base(
        excel_path: str,
        persist_dir: str = "./chroma_db",
        collection_name: str = "university_db",
        api_key: str = None  # Add API key as an argument
):
    """
    Checks if the Chroma DB exists.
    - If YES: Loads and returns the collection immediately.
    - If NO: Reads Excel, chunks text, embeds via Gemini, and saves to Chroma.
    """

    # ---------------------------------------------------------
    # 1. Check if DB exists (SKIP API CALLS IF TRUE)
    # ---------------------------------------------------------
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print(f"Database found at '{persist_dir}'. Skipping ingestion.")
        client = PersistentClient(path=persist_dir)
        return client.get_collection(collection_name)

    print(f"Database not found. Starting ingestion from {excel_path}...")

    # ---------------------------------------------------------
    # 2. Setup Gemini Client (ONLY IF NEEDED)
    # ---------------------------------------------------------
    # Try to get key from argument, or environment variable, or hardcoded string
    resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or "AIzaSyCXjChsmIUb4igBGrOxfvXUpxFSBeO2qh0"

    if not resolved_key:
        raise ValueError(
            "API Key is missing. Please set GEMINI_API_KEY environment variable or pass it to the function.")

    genai_client = genai.Client(api_key=resolved_key)

    # ---------------------------------------------------------
    # 3. Internal Helper Functions
    # ---------------------------------------------------------
    def clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def row_to_text(row: pd.Series) -> str:
        parts = [f"{col}: {val}" for col, val in row.items() if pd.notna(val) and str(val).strip() != ""]
        return " | ".join(parts)

    def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> List[str]:
        if len(text) <= max_chars: return [text]
        chunks, start = [], 0
        while start < len(text):
            end = start + max_chars
            if end < len(text) and text[end] != " ":
                space_back = text.rfind(" ", start, end)
                if space_back != -1: end = space_back
            chunks.append(text[start:end].strip())
            start = max(0, end - overlap)
        return chunks

    # ---------------------------------------------------------
    # 4. Process Excel
    # ---------------------------------------------------------
    df = pd.read_excel(excel_path, dtype=str).fillna("")

    docs, metadatas, ids = [], [], []

    for idx, row in df.iterrows():
        row_text = clean_text(row_to_text(row))
        chunks = chunk_text(row_text)

        for i, chunk in enumerate(chunks):
            docs.append(chunk)
            meta = {str(k): v for k, v in row.to_dict().items()}
            meta.update({"row_index": idx, "chunk_index": i, "source": os.path.basename(excel_path)})
            metadatas.append(meta)
            ids.append(f"{idx}_{i}_{uuid.uuid4().hex}")

    # ---------------------------------------------------------
    # 5. Generate Embeddings
    # ---------------------------------------------------------
    print(f"Generating embeddings for {len(docs)} chunks...")
    embeddings = []

    for text in docs:
        result = genai_client.models.embed_content(
            model="gemini-embedding-001",
            contents=text
        )
        embeddings.append(result.embeddings[0].values)

    # ---------------------------------------------------------
    # 6. Save to Chroma
    # ---------------------------------------------------------
    os.makedirs(persist_dir, exist_ok=True)
    client = PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print("Ingestion complete.")
    return collection