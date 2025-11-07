from typing import List
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import pdfplumber

# On-disk ChromaDB store (./.chroma)
_client = chromadb.Client(Settings(persist_directory=".chroma"))
_collection = _client.get_or_create_collection("healogue_docs")
_model = None

def _embedder():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

def _embed(texts: List[str]):
    return _embedder().encode(texts, normalize_embeddings=True).tolist()

def build_or_update_vectorstore_from_dataframe(df):
    try:
        docs = []
        for _, row in df.head(2000).iterrows():
            parts = [f"{col}: {row[col]}" for col in df.columns]
            docs.append("\n".join(parts))
        if not docs:
            return
        embeddings = _embed(docs)
        ids = [f"row-{i}" for i in range(len(docs))]
        _collection.upsert(documents=docs, embeddings=embeddings, ids=ids)
    except Exception:
        pass

def build_or_update_vectorstore_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        pages = []
        for p in pdf.pages:
            txt = p.extract_text() or ""
            if txt.strip():
                pages.append(txt.strip())
    if not pages:
        return
    embeddings = _embed(pages)
    ids = [f"pdf-{i}" for i in range(len(pages))]
    _collection.upsert(documents=pages, embeddings=embeddings, ids=ids)

def semantic_search_context(query: str, k: int = 4) -> List[str]:
    if _collection.count() == 0:
        return []
    qv = _embed([query])[0]
    res = _collection.query(query_embeddings=[qv], n_results=k)
    return res.get("documents", [[]])[0]
