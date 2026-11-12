# FinRAG: Retrieval-Augmented QA over SEC Filings

**A RAG pipeline for financial question answering on [FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench). It is evaluated on three separate axes (correctness, faithfulness and retrieval hit-rate) and tuned through measured improvement cycles.**

![Python](https://img.shields.io/badge/python-3.11+-blue) ![LangChain](https://img.shields.io/badge/LangChain-FAISS-1c3c3c) ![RAGAS](https://img.shields.io/badge/RAGAS-faithfulness-orange) ![Llama](https://img.shields.io/badge/Llama--3.3--70B-Nebius-black)

**Notebooks**: for the best reading experience, open them in nbviewer (full-page rendering) or Colab:

| Notebook | Read | Run |
|---|---|---|
| `financebench_rag.ipynb` | [![nbviewer](https://img.shields.io/badge/view-nbviewer-orange?logo=jupyter)](https://nbviewer.org/github/bloch-ai-engineering/fin-rag/blob/main/notebooks/financebench_rag.ipynb) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bloch-ai-engineering/fin-rag/blob/main/notebooks/financebench_rag.ipynb) |


## The problem

LLMs know what "working capital" means but not what Corning's working capital was in FY2022. FinanceBench asks exactly those questions over real 10-K/10-Q filings, and it is hard on purpose. We wanted to learn **which component to fix when results disappoint**: the retriever or the generator.

## Pipeline

```
FinanceBench PDFs ──► PyPDF ──► RecursiveCharacterTextSplitter (1000 / 150 overlap)
                                          │
                                          ▼
                       BAAI/bge-small-en-v1.5 embeddings ──► FAISS index
                                                                │
question ──► top-k similarity search ──► grounded prompt ──► Llama-3.3-70B (Nebius) ──► answer + citations
```

## Experiments

| Task | What we did |
|---|---|
| 1. Naive generation | 10 questions (5 domain-relevant, 5 novel) answered with **no retrieval**. Result: 70% partially correct, 30% wrong, 0 refusals. The model is confidently wrong on numbers (e.g. off by 15× on PayPal's working capital). |
| 3. Indexing | Chunked and embedded the source filings into FAISS, then spot-checked retrieval against the gold evidence pages. |
| 4–5. RAG vs. naive | A side-by-side comparison of the two, with a failure analysis (below). |
| 6. Evaluation | **Correctness** by LLM judge, **faithfulness** with RAGAS, **hit-rate@k** against the gold evidence documents. |
| 7. Improvement cycles | Sweeps over top-k (3 / 4 / 6) and temperature (0.1 / 0.3 / 0.7). |

## Results

**Baseline RAG** (k=4, T=0.1): faithfulness **0.95**, hit-rate@4 **70%**, correctness 30% partial, 0% fully correct.

| Config | Correct | Hit-rate |
|---|---|---|
| **k = 3** | **40%** | 70% |
| k = 4 | 30% | 70% |
| k = 6 | 30% | **90%** |
| T = 0.1 / 0.3 | 30% | – |
| T = 0.7 | 20% | – |

**What we learned:**
- **Retrieval is the bottleneck, not generation.** Faithfulness is near-perfect: the model sticks to its context. The context itself is often wrong or incomplete.
- **More context ≠ better answers.** k=6 finds the right document 90% of the time but answers worse than k=3. Extra chunks add noise.
- **Partial context silences uncertainty.** When the retriever returned a table with only 4 of JPM's 5 segments, RAG confidently picked the wrong one. That's harder to catch than a naive hallucination, because it comes with a citation.
- **RAG wins on exact-number extraction. Naive wins on high-level yes/no questions**, where RAG tends to refuse when retrieval misses.

## Project structure

```
├── notebooks/financebench_rag.ipynb   # full pipeline + written analysis, outputs included
├── results/
│   ├── 1_naive_generation.csv
│   ├── 2_naive_vs_rag.xlsx
│   ├── 3_rag_evaluation.xlsx          # correctness / faithfulness / hit-rate per question
│   └── 4_improvement_cycles.xlsx
└── docs/assignment/assignment_2_rag.pdf
```

## Run it

The notebook was built for Google Colab with a GPU, for the embeddings.

```bash
pip install -r requirements.txt
export NEBIUS_API_KEY=...        # or a Colab secret
jupyter notebook notebooks/financebench_rag.ipynb
```

Building the FAISS index from the filings is slow, so the notebook can download a pre-built index instead (see "Speed Up: Load Pre-built Vector Store").

## Context

This was an assignment for **Nebius Academy: AI Performance Engineering** (module "From AI Model to AI Agent", Assignment 2). The original brief is in [`docs/assignment/`](docs/assignment/).

## Authors

**Nathanael Bloch** · **Tomer Porat**
