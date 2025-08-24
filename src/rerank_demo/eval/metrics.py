from typing import List
import numpy as np
def _dcg(scores: List[float]) -> float:
    return sum((s / np.log2(i + 2)) for i, s in enumerate(scores))

def ndcg_at_k(binary_rels: List[int], k: int = 10) -> float:
    gains = binary_rels[:k]
    ideal = sorted(binary_rels, reverse=True)[:k]
    denom = _dcg(ideal) or 1.0
    return _dcg(gains) / denom

def mrr_at_k(binary_rels: List[int], k: int = 10) -> float:
    for i, rel in enumerate(binary_rels[:k], start=1):
        if rel:
            return 1.0 / i
    return 0.0
