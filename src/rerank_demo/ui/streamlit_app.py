import time
import streamlit as st
from rerank_demo.config import TOPK_RECALL, TOPK_SHOW
from rerank_demo.qdrant.client import get_client  
from rerank_demo.models.embedder import Embedder
from rerank_demo.models.reranker import CrossEncoderReranker
from rerank_demo.qdrant.search import retrieve_dense
st.set_page_config(page_title="Qdrant Two-Stage Retrieval", layout="wide")

@st.cache_resource
def bootstrap():
    client = get_client()
    emb = Embedder()
    ce = CrossEncoderReranker()
    return client, emb, ce

client, emb, ce = bootstrap()

st.title("Two-Stage Retrieval: Qdrant → Cross-Encoder Rerank")
q = st.text_input("Ask a question (FiQA examples work well):", "What is enterprise value in finance?")

if st.button("Search") and q.strip():
    t0 = time.time()
    qvec = emb.encode([q])[0]
    cands = retrieve_dense(client, qvec, topk=TOPK_RECALL)
    t1 = time.time()
    post = ce.rerank(q, cands)[:TOPK_SHOW]
    t2 = time.time()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Before (dense only)")
        for i, c in enumerate(cands[:TOPK_SHOW], 1):
            st.markdown(f"**{i}.** ({c['score']:.3f}) — {c['text'][:220]}…")

    with col2:
        st.subheader("After (cross-encoder reranked)")
        for i, c in enumerate(post, 1):
            st.markdown(f"**{i}.** ({c['rerank_score']:.3f}) — {c['text'][:220]}…")

    st.caption(f"Latency — recall: {(t1 - t0)*1000:.1f} ms, rerank: {(t2 - t1)*1000:.1f} ms, total: {(t2 - t0)*1000:.1f} ms")
