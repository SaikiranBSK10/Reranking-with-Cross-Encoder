import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION", "rerank_demo")

# Models
EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Pipeline params
TOPK_RECALL = int(os.getenv("TOPK_RECALL", "50"))
TOPK_SHOW = int(os.getenv("TOPK_SHOW", "10"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "256"))

# Dataset
DATA_DIR = os.getenv("DATA_DIR", "./datasets")
DATASET = os.getenv("DATASET", "fiqa")  
EVAL_LIMIT = int(os.getenv("EVAL_LIMIT", "200"))
