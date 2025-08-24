from typing import Dict, Any, List
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from qdrant_client import QdrantClient
from tqdm import tqdm
from uuid import uuid5, NAMESPACE_DNS  
from ..config import COLLECTION, BATCH_SIZE
from ..logging import get_logger

logger = get_logger(__name__)

def recreate_collection(client: QdrantClient, dim: int):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        logger.info("Collection %s already exists; recreating...", COLLECTION)
        client.delete_collection(collection_name=COLLECTION)
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    logger.info("Collection %s ready (dim=%d).", COLLECTION, dim)

def _to_point_id(doc_id: Any):
    """
    Convert BEIR doc_id to a Qdrant point ID:
    - numeric strings -> int
    - everything else -> deterministic UUIDv5
    """
    s = str(doc_id)
    if s.isdigit():
        return int(s)
    return str(uuid5(NAMESPACE_DNS, f"beir::{s}"))

def upsert_embeddings(client: QdrantClient, texts: List[str], ids: List[str], vectors: List[List[float]]):
    points = [
        PointStruct(
            id=_to_point_id(ids[i]),                     
            vector=vectors[i],
            payload={
                "doc_id": str(ids[i]),                  
                "text": texts[i],
            },
        )
        for i in range(len(ids))
    ]
    client.upsert(collection_name=COLLECTION, points=points)

def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def index_corpus(client: QdrantClient, embed_fn, corpus: Dict[str, Dict[str, str]]):
    items = list(corpus.items())
    logger.info("Indexing %d documents...", len(items))
    for batch in tqdm(list(chunked(items, BATCH_SIZE))):
        ids = [doc_id for doc_id, _ in batch]
        texts = [(meta.get("title","") + " " + meta.get("text","")).strip() for _, meta in batch]
        vecs = embed_fn(texts)  # returns List[List[float]]
        upsert_embeddings(client, texts, ids, vecs)
    logger.info("Indexing finished.")
