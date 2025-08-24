import argparse
import numpy as np
from tqdm import tqdm
from ..config import DATA_DIR, DATASET, TOPK_RECALL, TOPK_SHOW, EVAL_LIMIT
from ..logging import get_logger
from ..data.loader import load_beir
from ..qdrant.client import get_client
from ..models.embedder import Embedder
from ..qdrant.search import retrieve_dense
from ..models.reranker import CrossEncoderReranker
from ..eval.evaluate import evaluate_list

logger = get_logger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=EVAL_LIMIT, help="Number of queries to evaluate")
    args = parser.parse_args()

    corpus, queries, qrels = load_beir(DATASET, DATA_DIR, split="test")
    qids = list(queries.keys())[:args.limit]

    client = get_client()
    emb = Embedder()
    rerank = CrossEncoderReranker()

    ndcg_pre, ndcg_post, mrr_pre, mrr_post = [], [], [], []

    for qid in tqdm(qids):
        q = queries[qid]
        qvec = emb.encode([q])[0]
        cands = retrieve_dense(client, qvec, topk=TOPK_RECALL)
        pre = cands[:TOPK_SHOW]
        post = rerank.rerank(q, cands)[:TOPK_SHOW]

        res = evaluate_list(qid, pre, post, qrels)
        ndcg_pre.append(res["nDCG@10_pre"]); ndcg_post.append(res["nDCG@10_post"])
        mrr_pre.append(res["MRR@10_pre"]);   mrr_post.append(res["MRR@10_post"])

    print("\n=== Evaluation (avg over", len(qids), "queries) ===")
    print(f"nDCG@10  no-rerank: {np.mean(ndcg_pre):.4f}  | rerank: {np.mean(ndcg_post):.4f}")
    print(f"MRR@10   no-rerank: {np.mean(mrr_pre):.4f}   | rerank: {np.mean(mrr_post):.4f}")

if __name__ == "__main__":
    main()
