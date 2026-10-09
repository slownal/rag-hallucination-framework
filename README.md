# Sentence-Level Hallucination Detection & Tiered Abstention Framework for RAG

> **M.Tech Project** — *"Sentence-Level Detection and Tiered Abstention Framework for Reducing Hallucinations in Domain-Specific RAG Systems"*

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

A fully functional pipeline that detects hallucinations in LLM-generated medical answers **at the sentence level** and applies a **four-tier abstention policy** instead of a coarse binary accept/reject. Each generated sentence is independently verified against retrieved medical evidence using Natural Language Inference (NLI), then routed to one of four actions:

| Tier | Action | Condition |
|---|---|---|
|  **DISPLAY** | Show the sentence as-is | Entailment prob ≥ 0.70 |
|  **SUPPRESS** | Drop the sentence entirely | Contradiction prob ≥ 0.50 |
|  **RE_RETRIEVE** | Widen search & retry (up to 2×) | Entailment prob < 0.30 |
|  **FLAG** | Show with unverified warning | Borderline / inconclusive |

---

## Datasets

| Dataset | Source | Purpose |
|---|---|---|
| **PubMedQA** | [`qiaojin/PubMedQA`](https://huggingface.co/datasets/qiaojin/PubMedQA) | Retrieval corpus — 4,487 indexed PubMed abstract chunks |
| **MedHallu** | [`UTAustin-AIHealth/MedHallu`](https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu) | Hallucination detection benchmark — 1,000 correct + 1,000 hallucinated answer pairs |
| **MIRAGE** | [`gzxiong/MIRAGE`](https://github.com/gzxiong/MIRAGE) | RAG evaluation benchmark — MedQA, MedMCQA, PubMedQA, BioASQ, MMLU sub-benchmarks |

All datasets are downloaded **automatically** at runtime via the HuggingFace `datasets` library. No manual downloads required.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  Retriever (retriever.py)           │
│  sentence-transformers + cosine sim │
│  corpus: PubMedQA abstracts         │
└───────────────┬─────────────────────┘
                │ top-k evidence docs
                ▼
┌─────────────────────────────────────┐
│  Generator (generator.py)           │
│  MockGenerator | OpenAIGenerator    │
└───────────────┬─────────────────────┘
                │ draft answer
                ▼
┌─────────────────────────────────────┐
│  Sentence Splitter (sentence_splitter.py) │
│  NLTK / regex                       │
└───────────────┬─────────────────────┘
                │ claims[]
                ▼
┌─────────────────────────────────────┐
│  NLI Verifier (nli_verifier.py)     │
│  facebook/bart-large-mnli           │
│  entailment / neutral / contradiction│
└───────────────┬─────────────────────┘
                │ VerificationResult
                ▼
┌─────────────────────────────────────┐
│  Tiered Abstention (abstention.py)  │
│  DISPLAY / SUPPRESS / RE_RETRIEVE   │
│  / FLAG                             │
└───────────────┬─────────────────────┘
                │
                ▼
        Final Verified Response

Evaluation (evaluate.py):
  Hallucination Rate | Precision | Recall | FAR
  Benchmarks: MedHallu · MIRAGE
```

---

## Project Structure

```
rag_hallucination_framework/
├── config.py            # All thresholds, model names, dataset IDs
├── data_loader.py       # Dataset loaders: PubMedQA, MedHallu, MIRAGE
├── retriever.py         # Dense retrieval (sentence-transformers)
├── generator.py         # MockGenerator + OpenAIGenerator
├── sentence_splitter.py # NLTK / regex sentence boundary splitting
├── nli_verifier.py      # NLI-based claim verification (BART-MNLI)
├── abstention.py        # Four-tier abstention decision logic
├── pipeline.py          # End-to-end orchestrator with re-retrieval loop
├── evaluate.py          # Evaluation metrics computation
├── main.py              # CLI entry point
└── requirements.txt     # Python dependencies
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run offline demo (no API keys, no downloads needed)
```bash
cd rag_hallucination_framework
python main.py
```

### 3. Ask a single question
```bash
python main.py --query "What is the mechanism of action of metformin?"
```

### 4. Run evaluation on real benchmarks

```bash
# MedHallu — primary hallucination detection benchmark
python main.py --evaluate --dataset medhallu --max-examples 100

# MIRAGE — end-to-end RAG accuracy benchmark
python main.py --evaluate --dataset mirage --max-examples 100

# PubMedQA — medical QA evaluation
python main.py --evaluate --dataset pubmedqa --max-examples 100
```

### 5. Use OpenAI as the LLM backend
```bash
export OPENAI_API_KEY=sk-...
python main.py --backend openai --query "What treats hypertension?"
```

---

## Evaluation Metrics

| Metric | Formula | What it measures |
|---|---|---|
| **Hallucination Rate** | hallucinated sentences / total sentences | How hallucination-prone the raw generator is |
| **Precision** | TP / (TP + FP) | Of all SUPPRESS/FLAG decisions, how many were truly hallucinated |
| **Recall** | TP / (TP + FN) | Of all true hallucinations, how many did the system catch |
| **False Abstention Rate** | FP / (FP + TN) | How often the system incorrectly suppresses correct sentences |

---

## Configuration

All tunable parameters are in [`config.py`](rag_hallucination_framework/config.py):

```python
# Abstention thresholds (tune on a validation split)
ENTAILMENT_ACCEPT_THRESHOLD      = 0.70   # above → DISPLAY
CONTRADICTION_SUPPRESS_THRESHOLD = 0.50   # above → SUPPRESS
RETRIEVAL_INSUFFICIENT_THRESHOLD = 0.30   # below → RE_RETRIEVE

# Models
RETRIEVER_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL_NAME            = "facebook/bart-large-mnli"

# Retrieval
TOP_K_RETRIEVAL           = 3
MAX_RE_RETRIEVAL_ATTEMPTS = 2
```

---

## Citation

If you use this project, please cite the relevant datasets:

- **MedHallu**: [arXiv:2502.14302](https://arxiv.org/abs/2502.14302)
- **PubMedQA**: [arXiv:1909.06146](https://arxiv.org/abs/1909.06146)
- **MIRAGE**: [gzxiong/MIRAGE](https://github.com/gzxiong/MIRAGE)

---

## License

MIT
