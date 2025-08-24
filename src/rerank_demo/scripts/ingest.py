import traceback
from ..config import DATA_DIR, DATASET
from ..logging import get_logger
from ..data.loader import load_beir
from ..models.embedder import Embedder
from ..qdrant.client import get_client
from ..qdrant.index import recreate_collection, index_corpus

logger = get_logger(__name__)

def main():
    print("[ingest] start")
    try:
        logger.info("Loading BEIR dataset: %s", DATASET)
        corpus, _, _ = load_beir(DATASET, DATA_DIR, split="test")
        logger.info("Loaded corpus with %d docs", len(corpus))

        logger.info("Initializing embedder…")
        emb = Embedder()
        dim = emb.model.get_sentence_embedding_dimension()
        logger.info("Embedding dim=%d", dim)

        logger.info("Connecting to Qdrant…")
        client = get_client()                 
        client.get_collections()              
        logger.info("Qdrant is reachable.")

        logger.info("Recreating collection…")
        recreate_collection(client, dim)

        logger.info("Indexing corpus…")
        index_corpus(client, emb.encode, corpus)

        logger.info("Ingestion complete.")
        print("[ingest] done")
    except Exception as e:
        print("[ingest] ERROR:", e)
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
