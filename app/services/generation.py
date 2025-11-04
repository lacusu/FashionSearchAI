import json, os, re
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

    # color cue
    for c in ["red","blue","black","white","green","pink","beige","grey","gray","yellow","maroon","navy","brown","purple","orange"]:
        if c in tokens or c in q:
            if c in colour or c in name:
                cues.append(f"matches color “{c}”")

    # gender cue
    if any(w in tokens for w in ["men","man","male","boys"]):
        if any(w in name for w in ["men","man","male","boys"]):
            cues.append("fits men's segment")
    if any(w in tokens for w in ["women","woman","female","girls","ladies"]):
        if any(w in name for w in ["women","woman","female","girls","ladies"]):
            cues.append("fits women's segment")

    # item type cue
    types = ["shirt","t-shirt","tee","dress","jumpsuit","kurta","saree","jeans","shoe","sneaker","trouser","shorts","hoodie","sweater","jacket"]
    for t in types:
        if t in q and t in name:
            cues.append(f"type match “{t}”")

    # default semantic cue
    if not cues:
        cues.append("high semantic similarity to query")

    return "; ".join(cues)

def _fallback_generate(query, docs):
    return json.dumps({
        "query": query,
        "recommendations": [
            {
                "name": d["metadata"].get("name"),
                "brand": d["metadata"].get("brand"),
                "price": d["metadata"].get("price"),
                "colour": d["metadata"].get("colour"),
                "image": d["metadata"].get("image"),
                "reason": _compute_reason(query, d["metadata"], d["document"])
            }
            for d in docs[:3]
        ],
        "note": "Rule-based fallback (no OpenAI API key provided)"
    }, ensure_ascii=False, indent=2)

# optional LLM mode (kept for extra credit if you set OPENAI_API_KEY)
try:
    PROMPT = open(PROMPT_PATH, "r", encoding="utf-8").read()
except Exception:
    PROMPT = "You are a fashion expert. Return JSON only."

def generate(query: str, top_docs: list[dict]) -> str:
    if OPENAI_API_KEY:
        try:
            import openai
            openai.api_key = OPENAI_API_KEY
            context = json.dumps([
                {
                    "name": d["metadata"].get("name"),
                    "brand": d["metadata"].get("brand"),
                    "price": d["metadata"].get("price"),
                    "colour": d["metadata"].get("colour"),
                    "image": d["metadata"].get("image"),
                    "snippet": d["document"][:350]
                }
                for d in top_docs
            ], ensure_ascii=False)

            user_msg = f"Query: {query}\nContext: {context}\nReturn JSON only as instructed."
            resp = openai.ChatCompletion.create(
                model=os.getenv("GEN_MODEL", "gpt-4o-mini"),
                messages=[{"role": "system", "content": PROMPT},
                          {"role": "user", "content": user_msg}],
                temperature=0.2
            )
            return resp.choices[0].message["content"]
        except Exception as e:
            logger.warning(f"LLM generation failed; using fallback. Error: {e}")

    return _fallback_generate(query, top_docs)
