from typing import List, Dict, Any, Literal
from sentence_transformers import CrossEncoder
from ..config import RERANK_MODEL

class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        self.model = CrossEncoder(model_name)

    def score(self, query: str, passages: List[str]) -> List[float]:
        pairs = [(query, p) for p in passages]
        scores = self.model.predict(pairs)
        return [float(s) for s in scores]

    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        texts = [c["text"] for c in candidates]
        scores = self.score(query, texts)
        for s, c in zip(scores, candidates):
            c["rerank_score"] = s
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
