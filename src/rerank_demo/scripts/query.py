import argparse, time
from ..config import TOPK_RECALL, TOPK_SHOW
from ..logging import get_logger
from ..qdrant.client import get_client
from ..qdrant.search import retrieve_dense
from ..models.embedder import Embedder
from ..models.reranker import CrossEncoderReranker

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", required=True, help="Query text")
    parser.add_argument("--k", type=int, default=TOPK_RECALL, help="Recall candidates")
    args = parser.parse_args()

    client = get_client()
    emb = Embedder()
    rerank = CrossEncoderReranker()

    t0 = time.time()
    qvec = emb.encode([args.q])[0]
    cands = retrieve_dense(client, qvec, topk=args.k)
    t1 = time.time()
    post = rerank.rerank(args.q, cands)[:TOPK_SHOW]
    t2 = time.time()

    print("\n=== Before (dense only) ===")
    for i, c in enumerate(cands[:10], 1):
        print(f"{i:2d}. ({c['score']:.3f}) {c['text'][:140]}...")

    print("\n=== After (cross-encoder reranked) ===")
    for i, c in enumerate(post, 1):
        print(f"{i:2d}. ({c['rerank_score']:.3f}) {c['text'][:140]}...")

    print(f"\nLatency (single query) — "
          f"recall: {(t1 - t0)*1000:.1f} ms | "
          f"rerank: {(t2 - t1)*1000:.1f} ms | "
          f"total: {(t2 - t0)*1000:.1f} ms")

if __name__ == "__main__":
    main()
