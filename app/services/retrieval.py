import chromadb
from typing import Dict, Any
import numpy as np
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
    qvec = encode([query])[0]
    if hasattr(qvec, "tolist"):
        qvec = qvec.tolist()
    col = get_collection()
    res = col.query(
        query_embeddings=[qvec],
        n_results=k,
        include=["documents", "metadatas", "distances"],  # no ids
    )

    # Convert cosine distance -> similarity (1 - distance)
    dists = res.get("distances", [[]])[0]
    sims = [float(1.0 - d) for d in dists]
    res["similarities"] = [sims]  # keep same nesting shape as chroma
    return res
