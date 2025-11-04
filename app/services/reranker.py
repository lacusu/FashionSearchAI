import numpy as np
from app.services.embeddings import encode

def _cosine(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return b @ a  # (n,) sims

def rerank_with_scores(query, docs, top_k=3):
    qv = encode([query])[0]
    dv = encode(docs)
    sims = _cosine(qv, dv)

    # Keyword-aware boosts (explicit constraints)
    keywords = [w.lower() for w in query.split()]
    bonus = np.zeros_like(sims)
    for i, text in enumerate(docs):
      low = text.lower()
      for kw in keywords:
        if kw and kw in low:
          bonus[i] += 0.10

    final = sims + bonus
    order = np.argsort(final)[::-1][:top_k]
    return order.tolist(), final[order].tolist()

def rerank(query, docs, top_k=3):
    idxs, _ = rerank_with_scores(query, docs, top_k)
    return [docs[i] for i in idxs]
