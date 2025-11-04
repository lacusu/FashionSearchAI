import pandas as pd
import chromadb
from pathlib import Path

from app.services.preprocessing import build_chunks
from app.services.embeddings import encode
from app.utils.settings import DATA_CSV, CHROMA_DIR
from app.utils.logger import get_logger

logger = get_logger(__name__)
COLLECTION_NAME = "fashion_hybrid"

def main():
    logger.info("Loading dataset...")
    csv_path = Path(DATA_CSV)
    if not csv_path.exists():
        raise FileNotFoundError(f"DATA_CSV not found at {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    logger.info("Preprocessing and building chunks...")
    df = build_chunks(df)

    # Use hybrid chunk text
    chunks = df["chunk_hybrid"].tolist()

    logger.info("Encoding embeddings (this can take time on CPU)...")
    emb = encode(chunks)              # numpy array
    embeddings = emb.tolist()         # list of lists for Chroma

    logger.info("Creating Chroma collection...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    col = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    # Build metadata
    meta_cols = [c for c in ["name","brand","price","colour","image"] if c in df.columns]
    metadatas = df[meta_cols].fillna("NA").to_dict(orient="records")

    # IDs: prefer p_id if present
    if "p_id" in df.columns:
        ids = df["p_id"].astype(str).tolist()
    else:
        ids = [str(i) for i in range(len(chunks))]

    logger.info("Upserting vectors into Chroma with batching...")

    batch_size = 5000
    for i in range(0, len(chunks), batch_size):
        batch_ids = ids[i:i + batch_size]
        batch_emb = embeddings[i:i + batch_size]
        batch_docs = chunks[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]

        col.upsert(
            ids=batch_ids,
            embeddings=batch_emb,
            documents=batch_docs,
            metadatas=batch_meta
        )
        logger.info(f"Inserted: {i + len(batch_ids)}/{len(chunks)}")

    logger.info("Vector DB build complete.")

if __name__ == "__main__":
    main()
