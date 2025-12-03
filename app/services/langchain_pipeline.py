from typing import List, Dict, Any

from chromadb.app import settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma as LCChroma
from langchain_core.prompts import PromptTemplate

from app.utils.settings import CHROMA_DIR, EMB_MODEL, LC_COLLECTION, OPENAI_API_KEY, GEN_MODEL
from app.utils.logger import get_logger

logger = get_logger(__name__)

# LangChain embedding wrapper, reusing your SBERT model
_embedder = HuggingFaceEmbeddings(model_name=EMB_MODEL)


def get_vectorstore() -> LCChroma:
    """Attach LangChain to the existing persisted Chroma collection."""
    return LCChroma(
        collection_name=LC_COLLECTION,
        persist_directory=CHROMA_DIR,
        embedding_function=_embedder,
    )


def lc_search(query: str, k: int = 10) -> List[Dict[str, Any]]:
    """
    Search via LangChain's Chroma wrapper.
    Returns a list of dicts with metadata and similarity.
    """
    vs = get_vectorstore()
    docs_scores = vs.similarity_search_with_relevance_scores(query, k=k)

    results: List[Dict[str, Any]] = []
    for doc, score in docs_scores:
        meta = doc.metadata or {}
        results.append({
            "name": meta.get("name"),
            "brand": meta.get("brand"),
            "price": meta.get("price"),
            "colour": meta.get("colour"),
            "image": meta.get("image"),
            "similarity": float(score),  # LC uses relevance score (higher = better)
            "document": doc.page_content,
        })
    return results


# ---------- Generation layer (LangChain LLM + fallback) ----------

_GEN_PROMPT = PromptTemplate.from_template(
    """You are a fashion search assistant. Write ONLY valid JSON.

Query: "{query}"

You are given a list of retrieved products with similarity scores:

{context}

Return JSON of the form:
{{
  "query": "{query}",
  "final_answer": "one short paragraph explaining why these items match the query",
  "recommendations": [
    {{
      "name": "...",
      "brand": "...",
      "price": <float>,
      "colour": "...",
      "image": "<path_or_url>",
      "similarity": <float>,
      "rerank_score": <float>,
      "reason": "short justification based on color, type, gender, and semantic fit"
    }}
  ]
}}

Use at most 3 recommendations. Do not include any text outside the JSON.
"""
)


def lc_generate(query: str, ranked_items: List[Dict[str, Any]]) -> str:
    """
    If OPENAI_API_KEY is present, generate via LangChain ChatOpenAI.
    Otherwise, fall back to your existing rule-based JSON generator.
    """
    # No API key -> use your existing fallback generation
    if not OPENAI_API_KEY:
        from app.services.generation import generate as fallback_generate
        return fallback_generate(query, ranked_items)

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=GEN_MODEL, temperature=0.2)

        context = [
            {
                "name": it.get("name"),
                "brand": it.get("brand"),
                "price": it.get("price"),
                "colour": it.get("colour"),
                "image": it.get("image"),
                "similarity": it.get("similarity"),
                "rerank_score": it.get("rerank_score"),
            }
            for it in ranked_items[:6]
        ]

        prompt = _GEN_PROMPT.format(query=query, context=context)
        resp = llm.invoke(prompt)
        return resp.content

    except Exception as e:
        logger.warning(f"LangChain LLM failed, using fallback. Error: {e}")
        from app.services.generation import generate as fallback_generate
        return fallback_generate(query, ranked_items)
