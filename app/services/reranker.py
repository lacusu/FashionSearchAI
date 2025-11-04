import numpy as np
from app.services.embeddings import encode

def rerank(query, docs, top_k=3):
    # Embed query and documents
    qvec = encode([query])[0]
    doc_vecs = encode(docs)

    # Normalize for cosine similarity
    qvec = qvec / np.linalg.norm(qvec)
    doc_vecs = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)

    sims = doc_vecs @ qvec  # cosine similarity

    # Keyword-aware boosting
    keywords = [w.lower() for w in query.split()]
    bonus = np.zeros_like(sims)

    for i, text in enumerate(docs):
        low = text.lower()
        for kw in keywords:
            if kw in low:
                bonus[i] += 0.10  # small boost for explicit keyword match

    final_scores = sims + bonus

    # Sort and select top_k
    idxs = np.argsort(final_scores)[::-1][:top_k]
    return [docs[i] for i in idxs]
