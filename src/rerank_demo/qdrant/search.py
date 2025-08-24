from typing import List, Dict, Any
from ..config import COLLECTION, TOPK_RECALL
from qdrant_client import QdrantClient

def retrieve_dense(client: QdrantClient, qvec: List[float], topk: int = TOPK_RECALL) -> List[Dict[str, Any]]:
    hits = client.search(collection_name=COLLECTION, query_vector=qvec, limit=topk)
    return [{"id": h.payload["doc_id"], "text": h.payload["text"], "score": float(h.score)} for h in hits]
