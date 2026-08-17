import streamlit as st
import faiss
import pickle
import re
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Hybrid Search for RAG", layout="wide")


def simple_tokenize(text):
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    return tokens


@st.cache_resource
def load_everything():
    index = faiss.read_index("faiss_index.bin")
    with open("saved_data.pkl", "rb") as f:
        saved = pickle.load(f)
    corpus = saved["corpus"]
    passage_ids = saved["passage_ids"]
    queries_data = saved["queries_data"]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    tokenized_corpus = [simple_tokenize(corpus[pid]) for pid in passage_ids]
    bm25 = BM25Okapi(tokenized_corpus)
    return index, corpus, passage_ids, queries_data, model, bm25


with st.spinner("Loading models and data... (first time only)"):
    index, corpus, passage_ids, queries_data, model, bm25 = load_everything()

st.success("Ready! Everything loaded successfully.")


def bm25_search(query_text, top_k=5):
    tokenized_query = simple_tokenize(query_text)
    scores = bm25.get_scores(tokenized_query)
    id_score_pairs = list(zip(passage_ids, scores))
    ranked = sorted(id_score_pairs, key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]


def dense_search(query_text, top_k=5):
    query_embedding = model.encode(
        [query_text], convert_to_numpy=True, normalize_embeddings=True
    )
    scores, indices = index.search(query_embedding, top_k)
    scores = scores[0]
    indices = indices[0]
    results = []
    for position, score in zip(indices, scores):
        pid = passage_ids[position]
        results.append((pid, score))
    return results


def normalize_scores(scores):
    scores = np.array(scores)
    min_score = scores.min()
    max_score = scores.max()
    if max_score - min_score == 0:
        return np.zeros_like(scores)
    return (scores - min_score) / (max_score - min_score)


def hybrid_search(query_text, top_k=5, alpha=0.3):
    tokenized_query = simple_tokenize(query_text)
    bm25_scores_raw = bm25.get_scores(tokenized_query)
    query_embedding = model.encode(
        [query_text], convert_to_numpy=True, normalize_embeddings=True
    )
    dense_scores_raw, dense_indices = index.search(query_embedding, len(passage_ids))
    dense_scores_raw = dense_scores_raw[0]
    dense_indices = dense_indices[0]
    dense_scores_aligned = np.zeros(len(passage_ids))
    for position, score in zip(dense_indices, dense_scores_raw):
        dense_scores_aligned[position] = score
    bm25_norm = normalize_scores(bm25_scores_raw)
    dense_norm = normalize_scores(dense_scores_aligned)
    hybrid_scores = alpha * bm25_norm + (1 - alpha) * dense_norm
    id_score_pairs = list(zip(passage_ids, hybrid_scores))
    ranked = sorted(id_score_pairs, key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]


tab1, tab2 = st.tabs(["🔍 Search", "📊 Evaluation Dashboard"])

with tab1:
    st.header("Compare BM25, Dense, and Hybrid Search")
    query_text = st.text_input("Enter your search query:", value="what is rba")
    alpha = st.slider(
        "Hybrid alpha (0 = pure Dense, 1 = pure BM25):",
        min_value=0.0, max_value=1.0, value=0.3, step=0.1
    )
    top_k = st.slider("Number of results to show:", min_value=1, max_value=10, value=5)
    search_clicked = st.button("Search")

    if search_clicked and query_text.strip() != "":
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("BM25")
            results = bm25_search(query_text, top_k=top_k)
            for rank, (pid, score) in enumerate(results, start=1):
                st.markdown(f"**Rank {rank}** | Score: {score:.4f}")
                st.write(corpus[pid][:200] + "...")
                st.divider()
        with col2:
            st.subheader("Dense")
            results = dense_search(query_text, top_k=top_k)
            for rank, (pid, score) in enumerate(results, start=1):
                st.markdown(f"**Rank {rank}** | Score: {score:.4f}")
                st.write(corpus[pid][:200] + "...")
                st.divider()
        with col3:
            st.subheader(f"Hybrid (alpha={alpha})")
            results = hybrid_search(query_text, top_k=top_k, alpha=alpha)
            for rank, (pid, score) in enumerate(results, start=1):
                st.markdown(f"**Rank {rank}** | Score: {score:.4f}")
                st.write(corpus[pid][:200] + "...")
                st.divider()

with tab2:
    st.header("Evaluation Results")
    st.subheader("BM25 vs Dense vs Hybrid (alpha=0.5)")
    st.caption("Evaluated on 196 queries with ground-truth relevance labels, k=5")
    comparison_results = pd.read_csv("comparison_results.csv", index_col=0)
    st.dataframe(comparison_results)

    st.subheader("Hybrid Search: Alpha Sweep")
    st.caption("Testing different BM25/Dense weightings to find the best blend")
    alpha_results = pd.read_csv("alpha_sweep_results.csv", index_col=0)
    st.dataframe(alpha_results)

    st.subheader("MRR by Alpha (visual comparison)")
    mrr_row = alpha_results.loc["MRR"]
    st.bar_chart(mrr_row)

    st.info(
        "Key finding: Hybrid Search with alpha=0.2–0.3 (mostly Dense Retrieval "
        "with a modest BM25 contribution) outperformed both pure BM25 and pure "
        "Dense Retrieval across Precision@5, Recall@5, and MRR on this dataset."
    )
