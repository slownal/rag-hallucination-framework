"""
data_loader.py
--------------
Loads the benchmark datasets named on slide 15 (Dataset Used):
  - MedHallu   : UTAustin-AIHealth/MedHallu  (HuggingFace)
  - PubMedQA   : qiaojin/PubMedQA            (HuggingFace)
  - MIRAGE     : gzxiong/MIRAGE              (GitHub JSON)

Each loader:
  1. Tries to fetch the real dataset via the `datasets` library (HF) or
     requests (MIRAGE GitHub raw JSON).
  2. Falls back gracefully to a small synthetic sample if the download
     fails (network issue, no HF token, etc.) so the rest of the pipeline
     always has data to run against.

Schema emitted by all loaders:
  - Retrieval corpus  → List[Document]
  - Evaluation bench  → List[QAExample]
"""

import os
import json
import random
import urllib.request
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import config


# --------------------------------------------------------------------------
# COMMON DATA STRUCTURES
# --------------------------------------------------------------------------

@dataclass
class Document:
    """A single retrievable evidence passage (e.g. a PubMed abstract chunk)."""
    doc_id: str
    text: str
    source: str = "unknown"          # e.g. "pubmedqa", "mirage"
    metadata: Dict = field(default_factory=dict)


@dataclass
class QAExample:
    """
    One evaluation example: a medical question, ground-truth answer, and
    (for hallucination benchmarks) a ground-truth hallucination label.
    """
    qid: str
    question: str
    ground_truth_answer: Optional[str] = None
    is_hallucinated: Optional[bool] = None   # ground-truth label
    supporting_context: Optional[str] = None  # gold evidence passage, if any
    source_dataset: str = "synthetic"


# --------------------------------------------------------------------------
# SMALL SYNTHETIC FALLBACK CORPUS  (used when no real dataset can be loaded)
# --------------------------------------------------------------------------
_SYNTHETIC_CORPUS = [
    Document(
        doc_id="doc_001",
        text=(
            "Metformin is the first-line pharmacological treatment for "
            "type 2 diabetes mellitus. It works primarily by reducing "
            "hepatic glucose production and improving insulin sensitivity "
            "in peripheral tissues."
        ),
        source="pubmedqa_synthetic",
    ),
    Document(
        doc_id="doc_002",
        text=(
            "Common side effects of metformin include gastrointestinal "
            "disturbances such as diarrhea, nausea, and abdominal "
            "discomfort, particularly when treatment is first initiated."
        ),
        source="pubmedqa_synthetic",
    ),
    Document(
        doc_id="doc_003",
        text=(
            "Aspirin is a nonsteroidal anti-inflammatory drug (NSAID) that "
            "inhibits cyclooxygenase enzymes and is commonly used for pain "
            "relief, fever reduction, and antiplatelet therapy in "
            "cardiovascular disease prevention."
        ),
        source="pubmedqa_synthetic",
    ),
    Document(
        doc_id="doc_004",
        text=(
            "Hypertension is generally defined as a sustained systolic "
            "blood pressure of 130 mmHg or higher, or a diastolic blood "
            "pressure of 80 mmHg or higher, according to several major "
            "cardiology guidelines."
        ),
        source="pubmedqa_synthetic",
    ),
    Document(
        doc_id="doc_005",
        text=(
            "Amoxicillin is a beta-lactam antibiotic used to treat a range "
            "of bacterial infections, including respiratory tract "
            "infections, ear infections, and urinary tract infections."
        ),
        source="pubmedqa_synthetic",
    ),
]

_SYNTHETIC_QA_EXAMPLES = [
    QAExample(
        qid="synthetic_q1",
        question="What is the first-line treatment for type 2 diabetes?",
        ground_truth_answer="Metformin is the first-line treatment.",
        is_hallucinated=False,
        supporting_context=_SYNTHETIC_CORPUS[0].text,
        source_dataset="synthetic",
    ),
    QAExample(
        qid="synthetic_q2",
        question="What are the side effects of metformin?",
        ground_truth_answer="Gastrointestinal effects such as diarrhea and nausea.",
        is_hallucinated=False,
        supporting_context=_SYNTHETIC_CORPUS[1].text,
        source_dataset="synthetic",
    ),
    QAExample(
        qid="synthetic_q3",
        question="How does aspirin help prevent heart attacks?",
        ground_truth_answer="Aspirin inhibits platelet aggregation via cyclooxygenase inhibition.",
        is_hallucinated=False,
        supporting_context=_SYNTHETIC_CORPUS[2].text,
        source_dataset="synthetic",
    ),
]


# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------

def _try_import_datasets():
    """Returns the HuggingFace `datasets` module, or None if not installed."""
    try:
        import datasets as hf_datasets
        return hf_datasets
    except ImportError:
        return None


def _chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """
    Splits a long text into overlapping character-level chunks.
    Used to break long PubMed abstracts into retrievable passages.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE_CHARS
    overlap = overlap or config.CHUNK_OVERLAP_CHARS
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


# --------------------------------------------------------------------------
# 1. RETRIEVAL CORPUS  — PubMedQA abstracts  (HuggingFace: qiaojin/PubMedQA)
# --------------------------------------------------------------------------

def load_retrieval_corpus() -> List[Document]:
    """
    Loads the retrieval corpus from PubMedQA (qiaojin/PubMedQA) via the
    HuggingFace `datasets` library.

    PubMedQA schema (pqa_labeled / pqa_artificial configs):
      - pubid        : int   (PubMed ID)
      - question     : str
      - context      : dict  {"contexts": [str], "labels": [str], "meshes": [str], ...}
      - long_answer  : str   (expert answer)
      - final_decision : str  (yes / no / maybe)

    We index the `context.contexts` passages as retrievable documents.
    Each context string may be chunked if it exceeds config.CHUNK_SIZE_CHARS.

    Falls back to the 5-document synthetic corpus if download fails.
    """
    hf = _try_import_datasets()
    if hf is None:
        print(
            "[data_loader] WARNING: `datasets` package not installed. "
            "Falling back to synthetic corpus. Run: pip install datasets"
        )
        return list(_SYNTHETIC_CORPUS)

    try:
        print("[data_loader] Loading PubMedQA corpus from HuggingFace "
              f"('{config.HF_PUBMEDQA_ID}', pqa_labeled split) ...")
        ds = hf.load_dataset(
            config.HF_PUBMEDQA_ID,
            "pqa_labeled",
            split="train",
            trust_remote_code=True,
        )

        docs: List[Document] = []
        max_docs = config.MAX_CORPUS_DOCS

        for row in ds:
            pubid = str(row["pubid"])
            contexts = row["context"]["contexts"] if row["context"] else []
            for i, ctx in enumerate(contexts):
                if not ctx or not ctx.strip():
                    continue
                for j, chunk in enumerate(_chunk_text(ctx)):
                    doc_id = f"pubmed_{pubid}_{i}_{j}"
                    docs.append(Document(
                        doc_id=doc_id,
                        text=chunk.strip(),
                        source="pubmedqa",
                        metadata={"pubid": pubid, "context_idx": i, "chunk_idx": j},
                    ))
            if max_docs and len(docs) >= max_docs:
                docs = docs[:max_docs]
                break

        print(f"[data_loader] Loaded {len(docs)} PubMedQA corpus documents.")
        return docs if docs else list(_SYNTHETIC_CORPUS)

    except Exception as exc:
        print(
            f"[data_loader] WARNING: PubMedQA download failed ({exc}). "
            "Falling back to 5-document synthetic corpus."
        )
        return list(_SYNTHETIC_CORPUS)


# --------------------------------------------------------------------------
# 2. MEDHALLU EVALUATION EXAMPLES  (HuggingFace: UTAustin-AIHealth/MedHallu)
# --------------------------------------------------------------------------

def load_medhallu_examples(max_examples: int = None) -> List[QAExample]:
    """
    Loads MedHallu from HuggingFace (UTAustin-AIHealth/MedHallu).

    MedHallu schema (pqa_labeled / pqa_artificial configs):
      - Question                 : str
      - Knowledge                : list[str]  (supporting evidence passages)
      - Ground Truth             : str        (correct answer)
      - Difficulty Level         : str        (easy / medium / hard)
      - Hallucinated Answer      : str        (LLM-generated hallucination)
      - Category of Hallucination: str

    We yield TWO QAExample rows per item:
      - one with is_hallucinated=False  (ground-truth answer)
      - one with is_hallucinated=True   (hallucinated answer)
    This lets the verifier be scored on both correct and hallucinated outputs.

    Falls back to synthetic examples if download fails.
    """
    hf = _try_import_datasets()
    if hf is None:
        print(
            "[data_loader] WARNING: `datasets` package not installed. "
            "Falling back to synthetic MedHallu examples."
        )
        return list(_SYNTHETIC_QA_EXAMPLES)

    try:
        print("[data_loader] Loading MedHallu from HuggingFace "
              f"('{config.HF_MEDHALLU_ID}', pqa_labeled split) ...")
        ds = hf.load_dataset(
            config.HF_MEDHALLU_ID,
            "pqa_labeled",
            split="train",
            trust_remote_code=True,
        )

        examples: List[QAExample] = []
        max_ex = max_examples or config.MAX_MEDHALLU_EXAMPLES

        for i, row in enumerate(ds):
            question = row.get("Question", "").strip()
            ground_truth = row.get("Ground Truth", "").strip()
            hallucinated = row.get("Hallucinated Answer", "").strip()
            knowledge = row.get("Knowledge", [])
            difficulty = row.get("Difficulty Level", "unknown")
            category = row.get("Category of Hallucination", "unknown")

            # Build supporting context from Knowledge passages
            supporting = " ".join(knowledge) if knowledge else None

            # Ground-truth (non-hallucinated) example
            examples.append(QAExample(
                qid=f"medhallu_{i}_correct",
                question=question,
                ground_truth_answer=ground_truth,
                is_hallucinated=False,
                supporting_context=supporting,
                source_dataset="medhallu",
            ))

            # Hallucinated example (is_hallucinated=True)
            if hallucinated:
                examples.append(QAExample(
                    qid=f"medhallu_{i}_hallucinated",
                    question=question,
                    ground_truth_answer=hallucinated,
                    is_hallucinated=True,
                    supporting_context=supporting,
                    source_dataset="medhallu",
                ))

            if max_ex and len(examples) >= max_ex * 2:
                break

        print(f"[data_loader] Loaded {len(examples)} MedHallu examples "
              f"({sum(1 for e in examples if not e.is_hallucinated)} correct, "
              f"{sum(1 for e in examples if e.is_hallucinated)} hallucinated).")
        return examples if examples else list(_SYNTHETIC_QA_EXAMPLES)

    except Exception as exc:
        print(
            f"[data_loader] WARNING: MedHallu download failed ({exc}). "
            "Falling back to synthetic examples."
        )
        return list(_SYNTHETIC_QA_EXAMPLES)


# --------------------------------------------------------------------------
# 3. PUBMEDQA EVALUATION EXAMPLES  (HuggingFace: qiaojin/PubMedQA)
# --------------------------------------------------------------------------

def load_pubmedqa_examples(max_examples: int = None) -> List[QAExample]:
    """
    Loads PubMedQA labeled split for evaluation.

    Each PubMedQA example has:
      - pubid           : int
      - question        : str
      - context.contexts: list[str]   (supporting abstracts)
      - long_answer     : str         (expert answer)
      - final_decision  : yes/no/maybe

    We treat long_answer as the ground truth. No hallucinated counterpart
    is provided natively (unlike MedHallu), so is_hallucinated=False for all.
    """
    hf = _try_import_datasets()
    if hf is None:
        print("[data_loader] WARNING: `datasets` not installed; using synthetic examples.")
        return list(_SYNTHETIC_QA_EXAMPLES)

    try:
        print("[data_loader] Loading PubMedQA evaluation examples from HuggingFace ...")
        ds = hf.load_dataset(
            config.HF_PUBMEDQA_ID,
            "pqa_labeled",
            split="train",
            trust_remote_code=True,
        )

        examples: List[QAExample] = []
        max_ex = max_examples or config.MAX_PUBMEDQA_EXAMPLES

        for row in ds:
            pubid = str(row["pubid"])
            question = row.get("question", "").strip()
            long_answer = row.get("long_answer", "").strip()
            contexts = row["context"]["contexts"] if row.get("context") else []
            supporting = " ".join(contexts) if contexts else None

            examples.append(QAExample(
                qid=f"pubmedqa_{pubid}",
                question=question,
                ground_truth_answer=long_answer,
                is_hallucinated=False,
                supporting_context=supporting,
                source_dataset="pubmedqa",
            ))

            if max_ex and len(examples) >= max_ex:
                break

        print(f"[data_loader] Loaded {len(examples)} PubMedQA evaluation examples.")
        return examples if examples else list(_SYNTHETIC_QA_EXAMPLES)

    except Exception as exc:
        print(
            f"[data_loader] WARNING: PubMedQA eval load failed ({exc}). "
            "Falling back to synthetic examples."
        )
        return list(_SYNTHETIC_QA_EXAMPLES)


# --------------------------------------------------------------------------
# 4. MIRAGE EVALUATION EXAMPLES  (GitHub JSON: gzxiong/MIRAGE)
# --------------------------------------------------------------------------

def load_mirage_examples(max_examples: int = None) -> List[QAExample]:
    """
    Loads MIRAGE benchmark from the GitHub repository (gzxiong/MIRAGE).

    Real MIRAGE schema (benchmark.json at repo root) — a nested dict:
      {
        "medqa": {
          "0000": {
            "question": "...",
            "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
            "answer": "B"
          },
          "0001": { ... },
          ...
        },
        "pubmedqa": { ... },
        "bioasq":   { ... },
        ...
      }

    Each item is a multiple-choice question. We set ground_truth_answer to
    the FULL TEXT of the correct option (not just the letter). MIRAGE does
    not provide hallucinated pairs natively, so is_hallucinated=False for all.
    (Augment with get_synthetic_hallucinated_pair() in evaluate.py if needed.)

    Falls back to synthetic examples if the download fails.
    """
    MIRAGE_URL = "https://raw.githubusercontent.com/gzxiong/MIRAGE/main/benchmark.json"

    raw_data = None
    try:
        print(f"[data_loader] Fetching MIRAGE benchmark from: {MIRAGE_URL}")
        req = urllib.request.Request(MIRAGE_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw_data = json.loads(resp.read().decode("utf-8"))
        print("[data_loader] MIRAGE download successful.")
    except Exception as exc:
        print(
            f"[data_loader] WARNING: Could not load MIRAGE data ({exc}). "
            "Falling back to synthetic examples."
        )
        return list(_SYNTHETIC_QA_EXAMPLES)

    try:
        examples: List[QAExample] = []

        # benchmark.json is a dict keyed by sub-dataset name (medqa, pubmedqa, etc.)
        # Each value is a dict keyed by item ID ("0000", "0001", ...)
        for sub_dataset, items_dict in raw_data.items():
            if not isinstance(items_dict, dict):
                continue
            for item_id, item in items_dict.items():
                if not isinstance(item, dict):
                    continue

                question = item.get("question", "").strip()
                options = item.get("options", {})
                answer_key = item.get("answer", "").strip()

                # Map answer letter → full option text
                correct_text = options.get(answer_key, answer_key)

                # Build supporting context from all options (useful for retrieval)
                context_parts = [f"({k}) {v}" for k, v in options.items()]
                supporting = " | ".join(context_parts) if context_parts else None

                qid = f"mirage_{sub_dataset}_{item_id}"
                examples.append(QAExample(
                    qid=qid,
                    question=question,
                    ground_truth_answer=correct_text,
                    is_hallucinated=False,
                    supporting_context=supporting,
                    source_dataset=f"mirage_{sub_dataset}",
                ))

                if max_examples and len(examples) >= max_examples:
                    break
            if max_examples and len(examples) >= max_examples:
                break

        print(
            f"[data_loader] Loaded {len(examples)} MIRAGE examples "
            f"from sub-datasets: {list(raw_data.keys())}"
        )
        return examples if examples else list(_SYNTHETIC_QA_EXAMPLES)

    except Exception as exc:
        print(
            f"[data_loader] WARNING: MIRAGE parsing failed ({exc}). "
            "Falling back to synthetic examples."
        )
        return list(_SYNTHETIC_QA_EXAMPLES)


# --------------------------------------------------------------------------
# 5. UNIFIED PUBLIC LOADER FUNCTIONS  (called by pipeline.py / evaluate.py)
# --------------------------------------------------------------------------

def load_evaluation_examples(dataset_name: str = "medhallu") -> List[QAExample]:
    """
    Unified entry point for loading evaluation benchmarks.

    Args:
        dataset_name: one of {"medhallu", "pubmedqa", "mirage", "synthetic"}.

    Returns:
        List[QAExample] with is_hallucinated labels where available.
    """
    loaders = {
        "medhallu": load_medhallu_examples,
        "pubmedqa": load_pubmedqa_examples,
        "mirage":   load_mirage_examples,
    }

    if dataset_name == "synthetic":
        print("[data_loader] Using synthetic QA examples (development mode).")
        return list(_SYNTHETIC_QA_EXAMPLES)

    loader = loaders.get(dataset_name)
    if loader is None:
        raise ValueError(
            f"Unknown dataset_name '{dataset_name}'. "
            f"Choose from: {list(loaders.keys()) + ['synthetic']}"
        )

    return loader()


def get_synthetic_hallucinated_pair(example: QAExample) -> QAExample:
    """
    Helper used by evaluate.py: creates a crude synthetic hallucinated
    counterpart for any QAExample that doesn't have one natively
    (e.g. PubMedQA, MIRAGE).
    """
    corrupted_answer = (
        example.ground_truth_answer.replace("Metformin", "Insulin")
        if "Metformin" in (example.ground_truth_answer or "")
        else f"{example.ground_truth_answer} "
             "(Note: this claim is a synthetic corruption for testing.)"
    )
    return QAExample(
        qid=example.qid + "_hallucinated",
        question=example.question,
        ground_truth_answer=corrupted_answer,
        is_hallucinated=True,
        supporting_context=example.supporting_context,
        source_dataset=example.source_dataset,
    )


# --------------------------------------------------------------------------
# SMOKE TEST
# --------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("CORPUS TEST")
    corpus = load_retrieval_corpus()
    print(f"Loaded {len(corpus)} corpus documents.")
    if corpus:
        print(f"  First doc: [{corpus[0].source}] {corpus[0].text[:80]}...")

    print("\n" + "=" * 60)
    print("MEDHALLU TEST")
    medhallu = load_evaluation_examples("medhallu")
    print(f"Loaded {len(medhallu)} MedHallu examples.")
    if medhallu:
        print(f"  First: qid={medhallu[0].qid}, "
              f"is_hallucinated={medhallu[0].is_hallucinated}")

    print("\n" + "=" * 60)
    print("PUBMEDQA TEST")
    pubmedqa = load_evaluation_examples("pubmedqa")
    print(f"Loaded {len(pubmedqa)} PubMedQA examples.")

    print("\n" + "=" * 60)
    print("MIRAGE TEST")
    mirage = load_evaluation_examples("mirage")
    print(f"Loaded {len(mirage)} MIRAGE examples.")
