from typing import Dict, Any, List
from ..eval.metrics import ndcg_at_k, mrr_at_k

def evaluate_list(qid: str, pre: List[dict], post: List[dict], qrels: Dict[str, Dict[str, int]]):
    def is_rel(doc_id: str) -> int:
        return 1 if qrels.get(qid, {}).get(doc_id, 0) > 0 else 0

    y_pre = [is_rel(d["id"]) for d in pre]
    y_post = [is_rel(d["id"]) for d in post]

    return {
        "nDCG@10_pre": ndcg_at_k(y_pre, 10),
        "nDCG@10_post": ndcg_at_k(y_post, 10),
        "MRR@10_pre": mrr_at_k(y_pre, 10),
        "MRR@10_post": mrr_at_k(y_post, 10),
    }
