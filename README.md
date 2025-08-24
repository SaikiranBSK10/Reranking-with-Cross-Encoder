# Qdrant Two-Stage Retrieval Demo (Recall → Cross-Encoder Rerank)

This repo contains a minimal, industry-style structure for a **two-stage retrieval** system:
1) Dense recall with **Qdrant**
2) Rerank top-K with a **Cross-Encoder**

Default models:
- Retriever: `sentence-transformers/all-MiniLM-L6-v2`
- Reranker:  `cross-encoder/ms-marco-MiniLM-L-6-v2`

Dataset: **BEIR / FiQA** test split (downloaded automatically).

## Quickstart

### 0) Prereqs
- Python 3.10+ recommended
- Docker (for Qdrant)

### 1) Install deps
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Start Qdrant
```bash
docker compose up -d
```

### 3) Ingest BEIR FiQA into Qdrant
```bash
python -m src.rerank_demo.scripts.ingest
```

### 4) Try a single query
```bash
python -m src.rerank_demo.scripts.query --q "What is enterprise value in finance?"
```

### 5) Evaluate (nDCG@10 / MRR@10) on a small subset
```bash
python -m src.rerank_demo.scripts.eval_beir --limit 200
```

### 6) Run the Streamlit app
```bash
streamlit run src/rerank_demo/ui/streamlit_app.py
```
