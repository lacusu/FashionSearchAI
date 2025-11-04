from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.routers.search import router as search_router
from app.routers.recommend import router as recommend_router
from app.utils.settings import IMAGES_DIR

app = FastAPI(
    title="Fashion RAG Search API",
    description="Generative semantic fashion search system",
    version="1.0"
)

# UI Directory
UI_DIR = Path(__file__).resolve().parents[1] / "ui"

# Static mounts
app.mount("/static", StaticFiles(directory=UI_DIR / "static"), name="static")
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

# Home page
@app.get("/", response_class=HTMLResponse)
def home():
    return (UI_DIR / "index.html").read_text(encoding="utf-8")

# Routers
app.include_router(search_router)
app.include_router(recommend_router)
