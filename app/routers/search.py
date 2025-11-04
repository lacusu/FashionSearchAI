from fastapi import APIRouter, Query
from app.services import retrieval, reranker

router = APIRouter()

@router.get("/search")
def search(q: str = Query(..., description="User query"), k: int = 3):
    res = retrieval.search(q, k=10)
    docs = [
        {"document": d, "metadata": m}
        for d, m in zip(res["documents"][0], res["metadatas"][0])
    ]

    # Ensure plain strings for reranker
    doc_texts = [str(d["document"]) for d in docs]
    ranked_texts = reranker.rerank(q, doc_texts, top_k=k)

    ranked_docs = []
    for txt in ranked_texts:
        for d in docs:
            if str(d["document"]) == txt:
                ranked_docs.append(d)
                break

    out = [
        {
            "name": d["metadata"].get("name"),
            "brand": d["metadata"].get("brand"),
            "price": d["metadata"].get("price"),
            "colour": d["metadata"].get("colour"),
            "image": d["metadata"].get("image"),
        }
        for d in ranked_docs
    ]
    return {"query": q, "top_k": k, "results": out}
