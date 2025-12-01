from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma as LCChroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from app.utils.settings import CHROMA_DIR, EMB_MODEL, LC_COLLECTION, OPENAI_API_KEY
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Reuse the same embedding model you used to build the DB
_embedder = HuggingFaceEmbeddings(model_name=EMB_MODEL)

def get_vectorstore() -> LCChroma:
    # This attaches to your existing persisted Chroma collection
    return LCChroma(
        collection_name=LC_COLLECTION,
        persist_directory=CHROMA_DIR,
        embedding_function=_embedder,
    )

def lc_search(query: str, k: int = 10) -> Dict[str, Any]:
    """Search via LangChain VectorStore wrapper (reuses existing Chroma DB)."""
    vs = get_vectorstore()
    # returns list[(Document, score)]
    docs_scores = vs.similarity_search_with_relevance_scores(query, k=k)
    # Prepare a Search-layer-like response
    out = []
    for doc, score in docs_scores:
        meta = doc.metadata or {}
        out.append({
            "name": meta.get("name"),
            "brand": meta.get("brand"),
            "price": meta.get("price"),
            "colour": meta.get("colour"),
            "image": meta.get("image"),
            "similarity": float(score),        # LC returns similarity (higher=better)
            "rerank_score": float(score),      # keep field for UI compatibility
            "document": doc.page_content,
        })
    return {"query": query, "results": out}

# ----- Generation via LangChain (optional OpenAI) -----

_GEN_PROMPT = PromptTemplate.from_template(
    """You are a fashion search assistant. Write ONLY valid JSON.

Query: "{query}"

Use the retrieved products (with similarity scores) to produce a concise JSON:
{{
  "query": "{query}",
  "final_answer": "short one-paragraph rationale",
  "recommendations": [
    {{
      "name": "...",
      "brand": "...",
      "price": <float>,
      "colour": "...",
      "image": "<path_or_url>",
      "similarity": <float>,
      "rerank_score": <float>,
      "reason": "short justification (color/type/gender/semantic)"
    }}
  ]
}}

Keep it to the top 3 items, be precise, and never include markdown.
"""
)

def lc_generate(query: str, ranked_items: List[Dict[str, Any]]) -> str:
    """
    If OPENAI_API_KEY is present -> use ChatOpenAI via LangChain.
    Else -> produce the same rule-based JSON you already have.
    """
    # If no key, reuse your fallback
    if not OPENAI_API_KEY:
        from app.services.generation import generate as fallback_gen
        # generation.generate expects ranked_docs fields we already have
        return fallback_gen(query, ranked_items)

    # LLM path (OpenAI via LangChain)
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        context_json = [
            {
                "name": it.get("name"),
                "brand": it.get("brand"),
                "price": it.get("price"),
                "colour": it.get("colour"),
                "image": it.get("image"),
                "similarity": it.get("similarity"),
                "rerank_score": it.get("rerank_score"),
            } for it in ranked_items[:6]
        ]
        prompt = _GEN_PROMPT.format(query=query, context=context_json)
        resp = llm.invoke(prompt)
        return resp.content
    except Exception as e:
        logger.warning(f"LangChain LLM failed, using fallback. Error: {e}")
        from app.services.generation import generate as fallback_gen
        return fallback_gen(query, ranked_items)
