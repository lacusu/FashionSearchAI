from fastapi import APIRouter, Query
from app.utils.settings import USE_LANGCHAIN
from app.services import retrieval, reranker
from app.services import langchain_pipeline as lcp

router = APIRouter()

@router.get("/search")
def search(q: str = Query(..., description="User query"), k: int = 3):
    if USE_LANGCHAIN:
        # LC path: already returns similarity score
        res = lcp.lc_search(q, k=10)
        docs = res["results"]
        texts = [str(d.get("document", "")) for d in docs]
    else:
        # existing path
        res = retrieval.search(q, k=10)
        docs = [
            {"document": d, "metadata": m, "similarity": s}
            for d, m, s in zip(res["documents"][0], res["metadatas"][0], res["similarities"][0])
        ]
        texts = [str(d["document"]) for d in docs]

    # Our hybrid re-rank (fast, offline)
    idxs, scores = reranker.rerank_with_scores(q, texts, top_k=k)

    ranked_docs = []
    for i, sc in zip(idxs, scores):
        d = docs[i]
        meta = d.get("metadata", d)  # LC path stores fields at top-level
        ranked_docs.append({
            "name": meta.get("name"),
            "brand": meta.get("brand"),
            "price": meta.get("price"),
            "colour": meta.get("colour"),
            "image": meta.get("image"),
            "similarity": float(d.get("similarity", 0.0)),
            "rerank_score": float(sc),
        })

    return {"query": q, "top_k": k, "results": ranked_docs}
