from sentence_transformers import SentenceTransformer
from app.utils.settings import EMB_MODEL
from app.utils.logger import get_logger

logger = get_logger(__name__)
_model = None

def get_embedder():
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMB_MODEL}")
        _model = SentenceTransformer(EMB_MODEL, device="cpu")
    return _model

def encode(texts):
    model = get_embedder()
    return model.encode(texts, normalize_embeddings=True)
