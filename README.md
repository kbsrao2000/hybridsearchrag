# Hybrid Search for RAG (BM25 + Dense Retrieval)

A hybrid search engine built on the [MS MARCO v1.1](https://huggingface.co/datasets/microsoft/ms_marco) passage retrieval dataset, combining **BM25** (keyword/lexical search) with **dense embeddings** (semantic search) via FAISS. Includes a full evaluation pipeline and an interactive Streamlit demo.

## Overview

Traditional keyword search (BM25) is great at matching exact terms but misses semantic meaning. Dense retrieval (embeddings) captures meaning but can miss exact keyword matches (like rare terms, codes, or IDs). This project builds all three approaches — BM25-only, Dense-only, and a weighted Hybrid combination — and evaluates which performs best for passage retrieval.

**Key finding:** A hybrid blend with `alpha ≈ 0.2–0.3` (mostly dense retrieval, lightly boosted by BM25) outperformed both pure BM25 and pure dense retrieval on Precision@5, Recall@5, and MRR.

## How it works

1. **Corpus construction** — 3,000 queries from MS MARCO v1.1 are flattened into a corpus of ~24,772 unique passages, each linked back to the queries they're relevant to.
2. **BM25 index** — built with `rank_bm25` over simple lowercased/tokenized text.
3. **Dense index** — passages are embedded with `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim) and indexed with FAISS (`IndexFlatIP`, cosine similarity via normalized inner product).
4. **Hybrid search** — BM25 and dense scores are each min-max normalized, then combined:
   ```
   hybrid_score = alpha * bm25_score + (1 - alpha) * dense_score
   ```
5. **Evaluation** — all three methods are scored on 200 held-out queries using Precision@5, Recall@5, and Mean Reciprocal Rank (MRR), plus an alpha sweep (0.0 → 1.0) to find the optimal blend.

## Project structure

```
├── app.py                      # Streamlit demo app
├── jupyter.ipynb                # Full pipeline: data loading → indexing → evaluation
├── faiss_index.bin              # Saved FAISS dense index
├── saved_data.pkl               # Saved corpus, passage IDs, and query data
├── comparison_results.csv       # BM25 vs Dense vs Hybrid (alpha=0.5) results
├── alpha_sweep_results.csv      # Hybrid performance across alpha values
└── README.md
```

## Setup

```bash
pip install datasets rank_bm25 sentence-transformers faiss-cpu pandas numpy streamlit
```

## Usage

### Rebuild the pipeline from scratch
Run through `jupyter.ipynb` cell by cell. This will:
- Download MS MARCO v1.1
- Build the corpus, BM25 index, and dense/FAISS index
- Run evaluation and the alpha sweep
- Save `faiss_index.bin`, `saved_data.pkl`, and the result CSVs

### Run the interactive demo
Once the saved files exist, launch the Streamlit app:
```bash
streamlit run app.py
```

The app has two tabs:
- **🔍 Search** — enter a query and compare BM25, Dense, and Hybrid results side by side, with an adjustable alpha slider
- **📊 Evaluation Dashboard** — view the precomputed comparison table and alpha sweep results, including an MRR-by-alpha bar chart

## Evaluation metrics

- **Precision@5** — of the top 5 results, what fraction are actually relevant
- **Recall@5** — of all relevant passages, what fraction appear in the top 5
- **MRR (Mean Reciprocal Rank)** — averages `1 / rank of first relevant result` across queries; rewards ranking the correct passage as close to #1 as possible

## Results summary

| Method | MRR (approx.) |
|---|---|
| BM25 only | ~0.40 |
| Dense only | ~0.52 |
| Hybrid (alpha=0.2–0.3) | ~0.55 (best) |

## Tech stack

- [`datasets`](https://github.com/huggingface/datasets) — MS MARCO loading
- [`rank_bm25`](https://github.com/dorianbrown/rank_bm25) — BM25 keyword search
- [`sentence-transformers`](https://www.sbert.net/) — dense passage embeddings (`all-MiniLM-L6-v2`)
- [`faiss`](https://github.com/facebookresearch/faiss) — vector similarity search
- [`streamlit`](https://streamlit.io/) — interactive demo UI
