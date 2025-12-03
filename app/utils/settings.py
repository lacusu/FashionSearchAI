import os
from dotenv import load_dotenv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

DATA_CSV = str(ROOT_DIR / "data" / "FashionDatasetV2.csv")
CHROMA_DIR = str(ROOT_DIR / "vector_store" / "chroma")

# Images
IMAGES_DIR = str(ROOT_DIR / "data" / "images")
IMAGE_URL_PREFIX = "/images"

EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LC_COLLECTION = os.getenv("LC_COLLECTION", "fashion_hybrid")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4o-mini")

PROMPT_PATH = os.getenv("PROMPT_PATH", str(ROOT_DIR / "prompts" / "prompt_generation.txt"))
TOP_K_DEFAULT = int(os.getenv("TOP_K_DEFAULT", "3"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

def show_config_summary():
    print("=== FashionAI Settings ===")
    print(f"DATA_CSV     : {DATA_CSV}")
    print(f"IMAGES_DIR   : {IMAGES_DIR}")
    print(f"CHROMA_DIR   : {CHROMA_DIR}")
    print(f"EMB_MODEL    : {EMB_MODEL}")
    print(f"LC_COLLECTION: {LC_COLLECTION}")
    print(f"GEN_MODEL    : {GEN_MODEL}")
    print(f"OpenAI Key   : {'set' if OPENAI_API_KEY else 'not set'}")
    print("==========================")
