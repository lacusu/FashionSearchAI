from fastapi import APIRouter
from app.models.request_models import RecommendRequest
from app.services import reranker
from app.services import langchain_pipeline as lcp

router = APIRouter()

@router.post("/recommend")
def recommend(req: RecommendRequest):
    # 1) Retrieve via LangChain
    pool = lcp.lc_search(req.query, k=10)
    texts = [str(p.get("document", "")) for p in pool]

    # 2) Hybrid re-ranking
    idxs, scores = reranker.rerank_with_scores(req.query, texts, top_k=req.k)

    ranked = []
    for i, sc in zip(idxs, scores):
        p = pool[i]
        ranked.append({
            "name": p.get("name"),
            "brand": p.get("brand"),
            "price": p.get("price"),
            "colour": p.get("colour"),
            "image": p.get("image"),
            "similarity": float(p.get("similarity", 0.0)),
            "rerank_score": float(sc),
        })

    # 3) Generation via LangChain LLM or rule-based fallback
    gen_result = lcp.lc_generate(req.query, ranked)
    return {"query": req.query, "top_k": req.k, "generated": gen_result}
