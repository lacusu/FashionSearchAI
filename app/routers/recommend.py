from fastapi import APIRouter
from app.models.request_models import RecommendRequest
from app.services import retrieval, reranker, generation

router = APIRouter()

@router.post("/recommend")
def recommend(req: RecommendRequest):
    res = retrieval.search(req.query, k=10)

    docs = [
        {"document": d, "metadata": m}
        for d, m in zip(res["documents"][0], res["metadatas"][0])
    ]

    ranked_docs = []
    # Extract texts only for reranker to avoid ndarray issues
    docs_texts = [str(d["document"]) for d in docs]
    ranked_texts = reranker.rerank(req.query, docs_texts, top_k=req.k)
    for txt in ranked_texts:
        for d in docs:
            if str(d["document"]) == txt:
                ranked_docs.append(d)
                break

    gen_result = generation.generate(req.query, ranked_docs)

    return {"query": req.query, "top_k": req.k, "generated": gen_result}
