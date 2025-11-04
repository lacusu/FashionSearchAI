import json, os, re
from collections import defaultdict
from app.utils.settings import OPENAI_API_KEY, PROMPT_PATH
from app.utils.logger import get_logger

logger = get_logger(__name__)

def _compute_reason(query: str, meta: dict, snippet: str) -> str:
    q = query.lower()
    name = (meta.get("name") or "").lower()
    brand = (meta.get("brand") or "").lower()
    colour = (meta.get("colour") or "").lower()
    tokens = set(re.findall(r"[a-z]+", q))
    cues = []
    # colors
    for c in ["red","blue","black","white","green","pink","beige","grey","gray","yellow","maroon","navy","brown","purple","orange"]:
        if c in q and (c in colour or c in name):
            cues.append(f"matches color “{c}”")
    # gender
    if any(w in tokens for w in ["men","man","male","boys"]) and any(w in name for w in ["men","man","male","boys"]):
        cues.append("fits men's segment")
    if any(w in tokens for w in ["women","woman","female","girls","ladies"]) and any(w in name for w in ["women","woman","female","girls","ladies"]):
        cues.append("fits women's segment")
    # type
    types = ["shirt","t-shirt","tee","dress","jumpsuit","kurta","saree","jeans","shoe","sneaker","trouser","shorts","hoodie","sweater","jacket","running"]
    for t in types:
        if t in q and t in name:
            cues.append(f"type match “{t}”")
    if not cues:
        cues.append("high semantic similarity to query")
    return "; ".join(cues)

def _diversify(items, max_per_brand=2):
    # simple brand diversity: cap repeats
    seen = defaultdict(int)
    out = []
    for it in items:
        b = (it["brand"] or "NA").lower()
        if seen[b] < max_per_brand:
            out.append(it)
            seen[b] += 1
    return out


# This is a simple rule-based fallback generator
def _fallback_generate(query, ranked_docs):
    # ranked_docs items contain: name, brand, price, colour, image, similarity, rerank_score, boosts?
    diversified = _diversify(ranked_docs)
    summary = f"Top picks for “{query}” selected by semantic relevance, explicit keyword boosts, and brand diversity."
    return json.dumps({
        "query": query,
        "final_answer": summary,
        "recommendations": [
            {
                "name": d.get("name"),
                "brand": d.get("brand"),
                "price": d.get("price"),
                "colour": d.get("colour"),
                "image": d.get("image"),
                "similarity": d.get("similarity"),
                "rerank_score": d.get("rerank_score"),
                "reason": _compute_reason(query, d, "")  # pass meta-like dict
            } for d in diversified[:3]
        ],
        "note": "Rule-based fallback (no OpenAI API key provided)"
    }, ensure_ascii=False, indent=2)

try:
    PROMPT = open(PROMPT_PATH, "r", encoding="utf-8").read()
except Exception:
    PROMPT = "You are a fashion expert. Return JSON only."

def generate(query: str, ranked_docs: list[dict]) -> str:
    # ranked_docs already includes metadata and scoring
    if OPENAI_API_KEY:
        try:
            import openai
            openai.api_key = OPENAI_API_KEY
            context = json.dumps(ranked_docs[:6], ensure_ascii=False)
            user_msg = f"Query: {query}\nContext: {context}\nReturn JSON as instructed."
            resp = openai.ChatCompletion.create(
                model=os.getenv("GEN_MODEL", "gpt-4o-mini"),
                messages=[{"role":"system", "content":PROMPT},
                          {"role":"user", "content":user_msg}],
                temperature=0.2
            )
            return resp.choices[0].message["content"]
        except Exception as e:
            logger.warning(f"LLM generation failed; using fallback. Error: {e}")
    return _fallback_generate(query, ranked_docs)
