"""
config.py
---------
Single place that holds every tunable setting used across the pipeline
(model names, thresholds, file paths). Centralising these means that when
the real datasets (MIRAGE, RAGCare-QA, RAGTruth, MedHallu, PubMedQA) are
available, you only need to change values here rather than hunting
through every module.

Nothing in this file requires the datasets to run — it is pure
configuration.
"""

import os


# --------------------------------------------------------------------------
# 1. MODEL NAMES
# --------------------------------------------------------------------------
# Sentence-embedding model used by the Retriever to turn documents / queries
# into vectors for similarity search. 'all-MiniLM-L6-v2' is small, fast and
# works fine on CPU -- good for prototyping before you scale up.
RETRIEVER_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Natural Language Inference (NLI) model used by the Sentence-Level
# Verification Module. Given a (premise=evidence, hypothesis=generated
# sentence) pair, it outputs probabilities for {entailment, neutral,
# contradiction}. This is the core "does the evidence support this claim"
# check described in your slide 12 (System Architecture).
NLI_MODEL_NAME = "facebook/bart-large-mnli"

# Name / identifier of the answer-generation LLM. In the demo pipeline this
# is not actually called over the network (see generator.py) -- it is left
# swappable so you can later point it at OpenAI, Anthropic, or a local HF
# causal LM once you decide which backbone your project will use.
GENERATOR_MODEL_NAME = "gpt-4o-mini"  # TODO: replace with your chosen LLM


# --------------------------------------------------------------------------
# 2. TIERED ABSTENTION THRESHOLDS
# --------------------------------------------------------------------------
# These thresholds implement the "Display / Suppress / Re-retrieve / Flag"
# decision policy from slide 11 (Module Identification) and slide 12
# (System Architecture: "Confidence check -> Entailed or not").
#
# entailment_prob is the probability mass the NLI model assigns to the
# "entailment" label for the best-matching evidence chunk.
# contradiction_prob is the probability mass assigned to "contradiction".
#
#   entailment_prob >= ENTAILMENT_ACCEPT_THRESHOLD        -> DISPLAY
#   contradiction_prob >= CONTRADICTION_SUPPRESS_THRESHOLD -> SUPPRESS
#   entailment_prob < RETRIEVAL_INSUFFICIENT_THRESHOLD     -> RE_RETRIEVE
#   otherwise (borderline / neutral)                       -> FLAG
#
# These are starting points from common NLI-verification literature
# (e.g. RAGTruth, SelfCheckGPT use similar cutoffs); tune them once you
# have validation data.
ENTAILMENT_ACCEPT_THRESHOLD = 0.70
CONTRADICTION_SUPPRESS_THRESHOLD = 0.50
RETRIEVAL_INSUFFICIENT_THRESHOLD = 0.30

# Maximum number of times the pipeline will attempt re-retrieval for a
# single unsupported sentence before giving up and flagging it instead.
MAX_RE_RETRIEVAL_ATTEMPTS = 2


# --------------------------------------------------------------------------
# 3. RETRIEVAL SETTINGS
# --------------------------------------------------------------------------
# Number of top documents/chunks to retrieve per query.
TOP_K_RETRIEVAL = 3

# Chunk size (in characters) used when splitting long source documents into
# retrievable passages. Only relevant once you load real corpora.
CHUNK_SIZE_CHARS = 500
CHUNK_OVERLAP_CHARS = 50


# --------------------------------------------------------------------------
# 4. DATASET PATHS  (all TODOs -- no dataset is bundled with this code)
# --------------------------------------------------------------------------
# Fill these in once you download the datasets mentioned on slide 15.
# Until then, data_loader.py falls back to small synthetic samples so the
# rest of the pipeline can be developed and unit-tested independently.
DATA_DIR = os.environ.get("RAG_HALLU_DATA_DIR", "./data")

# Paths where downloaded datasets are cached
MEDHALLU_PATH = os.path.join(DATA_DIR, "medhallu")
MIRAGE_PATH = os.path.join(DATA_DIR, "mirage")
RAGCARE_QA_PATH = os.path.join(DATA_DIR, "ragcare_qa")
PUBMEDQA_PATH = os.path.join(DATA_DIR, "pubmedqa")

# --------------------------------------------------------------------------
# 5. HUGGING FACE DATASET IDs (used by data_loader.py via `datasets` library)
# --------------------------------------------------------------------------
# These are the official HuggingFace dataset identifiers for streaming/download.
HF_MEDHALLU_ID = "UTAustin-AIHealth/MedHallu"
HF_PUBMEDQA_ID = "qiaojin/PubMedQA"

# MIRAGE benchmark data (from GitHub, stored as JSON files)
MIRAGE_GITHUB_URL = "https://raw.githubusercontent.com/gzxiong/MIRAGE/main/data"

# Maximum samples to load from each dataset (None = load all)
# Reduce these during development to speed up iteration
MAX_CORPUS_DOCS = None          # e.g. 5000 to limit PubMedQA corpus
MAX_MEDHALLU_EXAMPLES = None    # e.g. 500 for faster evaluation
MAX_PUBMEDQA_EXAMPLES = None    # e.g. 500 for faster evaluation


# --------------------------------------------------------------------------
# 6. MISC
# --------------------------------------------------------------------------
RANDOM_SEED = 42
DEVICE = "cpu"  # change to "cuda" if a GPU is available in your environment
