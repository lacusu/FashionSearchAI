from fastapi import APIRouter, Query
from app.services import retrieval, reranker

router = APIRouter()

@router.get("/search")
def search(q: str = Query(..., description="User query"), k: int = 3):
    # Retrieve a wider pool (evidence-stage)
    res = retrieval.search(q, k=10)
    docs = [
        {"document": d, "metadata": m, "similarity": s}
        for d, m, s in zip(res["documents"][0], res["metadatas"][0], res["similarities"][0])
    ]

    # Pure retrieval+re-rank view (no synthesis)
    doc_texts = [str(d["document"]) for d in docs]
    ranked_ix, ranked_scores = reranker.rerank_with_scores(q, doc_texts, top_k=k)

    ranked_docs = []
    for idx, score in zip(ranked_ix, ranked_scores):
        d = docs[idx]
        ranked_docs.append({
            "name": d["metadata"].get("name"),
            "brand": d["metadata"].get("brand"),
            "price": d["metadata"].get("price"),
            "colour": d["metadata"].get("colour"),
            "image": d["metadata"].get("image"),
            "similarity": round(float(d["similarity"]), 4),
            "rerank_score": round(float(score), 4),
        })

    return {"query": q, "top_k": k, "results": ranked_docs}
