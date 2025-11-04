# Fashion RAG Search System

A production-style Retrieval-Augmented Generation (RAG) search application for fashion product recommendations.

## System Overview

This project implements a full generative semantic search workflow:

1. **Embedding Layer**
   * Dataset preprocessing and hybrid chunk strategy
   * SentenceTransformers model: `all-MiniLM-L6-v2`
   * Vector database: ChromaDB

2. **Search Layer**
   * Vector similarity search (cosine)
   * Cross-encoder reranking: `ms-marco-MiniLM-L-6-v2`
   * Query caching to reduce latency

3. **Generation Layer**
   * JSON-structured responses
   * LLM generation when API key is available
   * Rule-based fallback for local runs

## Architecture
User → UI (index.html) → FastAPI → Retrieval → Reranking → Generation → Result
↓
ChromaDB


## Quickstart

Install dependencies:
pip install -r requirements.txt

Prepare vector DB:
python scripts/build_db.py

Run the service:
uvicorn app.main:app --reload

Open browser:
http://localhost:8000/

## Screenshot Automation

Run server, then:
python scripts/generate_screenshots.py

Files saved in:
docs/screenshots/

## Testing Queries Used

1. Red party dress
2. Blue denim shirt men
3. Black running shoes

## Project Structure
app/
routers/
services/
models/
utils/
ui/
data/raw/
docs/screenshots/
scripts/
prompts/


## Notes

* For local execution without an OpenAI API Key, rule-based fallback logic is used.
* When using an API Key, the system generates high-quality JSON recommendations.

---

This project demonstrates the application of semantic search, hybrid chunking, vector similarity retrieval, reranking, and generative AI in a cohesive fashion recommendation system.



