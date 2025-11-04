import chromadb
from typing import Dict, Any
from app.services.embeddings import encode
from app.utils.settings import CHROMA_DIR
from app.utils.logger import get_logger

logger = get_logger(__name__)
client = chromadb.PersistentClient(path=CHROMA_DIR)
COLLECTION_NAME = "fashion_hybrid"

def get_collection():
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        logger.info("Collection not found. Please run scripts/build_db.py first.")
        return client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

def search(query: str, k=10) -> Dict[str, Any]:
    # Ensure list[float], not numpy.ndarray
    qvec = encode([query])[0]
    if hasattr(qvec, "tolist"):
        qvec = qvec.tolist()

    col = get_collection()
    res = col.query(
        query_embeddings=[qvec],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return res
