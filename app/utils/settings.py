import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
# Dataset + Chroma persistence paths
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_CSV = str(BASE_DIR / "data" / "FashionDatasetV2.csv")
CHROMA_DIR = str(BASE_DIR / "vector_store" / "chroma")

# Model selections
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Prompt for generation
PROMPT_PATH = str(BASE_DIR / "prompts" / "fashion_expert_prompt.txt")

# Optional API Key for GPT-based generation
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# Cache directory
CACHE_DIR = str(BASE_DIR / "cache")

# Images
IMAGES_DIR = str(BASE_DIR / "data" / "images")
IMAGE_URL_PREFIX = "/images"