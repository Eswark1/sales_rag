"""
RAG engine: embed free-text columns → vector search → return top-k snippets.
Indexes: activities.summary, leads.notes, opportunities.notes
"""

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from app.config import settings
from app.db import get_conn

COLLECTION_NAME = "sales_text"

_ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

_client: chromadb.PersistentClient | None = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=settings.chroma_path)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_ef,
        )
    return _collection


def build_index(force: bool = False) -> int:
    """
    Index all free-text rows. Safe to call on startup; skips if already indexed
    unless force=True.
    """
    col = _get_collection()
    if col.count() > 0 and not force:
        return col.count()

    sources = [
        ("SELECT 'activity-' || activity_id, summary, "
         "'entity:'||entity_type||':'||entity_id||' rep:'||rep_id||' type:'||activity_type "
         "FROM activities WHERE summary IS NOT NULL AND summary != ''"),
        ("SELECT 'lead-' || lead_id, notes, "
         "'company:'||company_name||' status:'||status||' source:'||source "
         "FROM leads WHERE notes IS NOT NULL AND notes != ''"),
        ("SELECT 'opp-' || opportunity_id, notes, "
         "'stage:'||stage||' value:'||estimated_value "
         "FROM opportunities WHERE notes IS NOT NULL AND notes != ''"),
    ]

    ids, docs, metas = [], [], []
    with get_conn() as conn:
        for sql in sources:
            for row_id, text, meta_str in conn.execute(sql).fetchall():
                ids.append(str(row_id))
                docs.append(text)
                metas.append({"source": meta_str})

    if ids:
        col.upsert(ids=ids, documents=docs, metadatas=metas)
    return len(ids)


def retrieve(question: str, top_k: int = 5) -> list[str]:
    col = _get_collection()
    if col.count() == 0:
        build_index()
    results = col.query(query_texts=[question], n_results=top_k)
    snippets = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        snippets.append(f"[{meta.get('source','')}] {doc}")
    return snippets
