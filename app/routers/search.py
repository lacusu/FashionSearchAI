from fastapi import APIRouter, Query
from app.services import reranker
from app.services import langchain_pipeline as lcp

router = APIRouter()

@router.get("/search")
def search(q: str = Query(..., description="User query"), k: int = 3):
    # 1) Retrieve via LangChain's Chroma wrapper
    pool = lcp.lc_search(q, k=10)
    texts = [str(p.get("document", "")) for p in pool]

    # 2) Hybrid re-ranking (cosine + keyword boosts)
    idxs, scores = reranker.rerank_with_scores(q, texts, top_k=k)

    ranked_docs = []
    for i, sc in zip(idxs, scores):
        p = pool[i]
        ranked_docs.append({
            "name": p.get("name"),
            "brand": p.get("brand"),
            "price": p.get("price"),
            "colour": p.get("colour"),
            "image": p.get("image"),
            "similarity": float(p.get("similarity", 0.0)),
            "rerank_score": float(sc),
        })

    return {"query": q, "top_k": k, "results": ranked_docs}
