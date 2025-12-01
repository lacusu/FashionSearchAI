from fastapi import APIRouter
from app.models.request_models import RecommendRequest
from app.utils.settings import USE_LANGCHAIN
from app.services import retrieval, reranker, generation
from app.services import langchain_pipeline as lcp

router = APIRouter()

@router.post("/recommend")
def recommend(req: RecommendRequest):
    if USE_LANGCHAIN:
        res = lcp.lc_search(req.query, k=10)
        pool = res["results"]
        texts = [str(p.get("document", "")) for p in pool]
    else:
        res = retrieval.search(req.query, k=10)
        pool = [
            {"document": d, "metadata": m, "similarity": s}
            for d, m, s in zip(res["documents"][0], res["metadatas"][0], res["similarities"][0])
        ]
        texts = [str(p["document"]) for p in pool]

    idxs, scores = reranker.rerank_with_scores(req.query, texts, top_k=req.k)

    ranked = []
    for i, sc in zip(idxs, scores):
        p = pool[i]
        meta = p.get("metadata", p)
        ranked.append({
            "name": meta.get("name"),
            "brand": meta.get("brand"),
            "price": meta.get("price"),
            "colour": meta.get("colour"),
            "image": meta.get("image"),
            "similarity": float(p.get("similarity", 0.0)),
            "rerank_score": float(sc),
        })

    # Generation via LC LLM if key, else your fallback
    gen_result = lcp.lc_generate(req.query, ranked)
    return {"query": req.query, "top_k": req.k, "generated": gen_result}
