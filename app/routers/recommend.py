from fastapi import APIRouter
from app.models.request_models import RecommendRequest
from app.services import retrieval, reranker, generation

router = APIRouter()

@router.post("/recommend")
def recommend(req: RecommendRequest):
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
        ranked.append({
            "name": p["metadata"].get("name"),
            "brand": p["metadata"].get("brand"),
            "price": p["metadata"].get("price"),
            "colour": p["metadata"].get("colour"),
            "image": p["metadata"].get("image"),
            "similarity": round(float(p["similarity"]), 4),
            "rerank_score": round(float(sc), 4),
        })

    gen_result = generation.generate(req.query, ranked)
    return {"query": req.query, "top_k": req.k, "generated": gen_result}
