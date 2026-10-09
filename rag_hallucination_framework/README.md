# Sentence-Level Hallucination Detection and Tiered Abstention Framework

Code scaffold for the M.Tech project described in your Review-II slides:
*"Sentence-Level Detection and Tiered Abstention Framework for Reducing
Hallucinations in Domain-Specific RAG Systems."*

## Status: ~60% complete, dataset-independent

Every module below is **fully implemented and runnable** — nothing is a
placeholder function that just returns dummy values. The only things
missing are the pieces that genuinely cannot exist without your real
datasets (MIRAGE, RAGCare-QA, RAGTruth, MedHallu, PubMedQA):

- Real corpus loading/parsing (currently a 5-document synthetic medical
  corpus stands in).
- Real benchmark loading/parsing (currently 3 synthetic QA pairs, plus a
  crude synthetic "hallucinated twin" for each, stand in for MedHallu).
- Threshold tuning (starting values are provided in `config.py`, based on
  common cutoffs in the literature you cited — tune once you have a
  validation split).

Every place this applies is marked with a `# TODO` comment and an
explanation of exactly what real parsing logic needs to go there and
what the target schema looks like.

---

## How the files map to your slides

| Slide | Concept | File(s) |
|---|---|---|
| Slide 11: Module Identification | Medical Document Retrieval Module | `retriever.py` |
| Slide 11 | Response Generation Module | `generator.py` |
| Slide 11 | Sentence-Level Verification Module | `sentence_splitter.py` + `nli_verifier.py` |
| Slide 11 | Tiered Abstention Module | `abstention.py` |
| Slide 11 | Evaluation Module | `evaluate.py` |
| Slide 12: System Architecture (the whole flowchart) | End-to-end wiring | `pipeline.py` |
| Slide 15: Dataset Used | Dataset loading (with synthetic fallback) | `data_loader.py` |
| Slide 10: Objectives (metrics) | Hallucination rate / precision / recall / FAR | `evaluate.py` |
| — | Configuration / thresholds | `config.py` |
| — | Runnable demo | `main.py` |

---

## File-by-file explanation

### `config.py`
Single source of truth for every tunable value: model names (embedding
model, NLI model, LLM name), the three abstention thresholds, retrieval
settings (top-k, chunk size), and dataset paths. Nothing here needs a
dataset to work — it's pure configuration. When you get real data, you
only edit the path constants at the bottom of this file; no other file
needs to change.

### `data_loader.py`
Loads (a) the retrieval corpus and (b) benchmark QA examples used for
evaluation. For each, it first checks whether a real dataset exists at
the configured path; if not, it prints a warning and returns a small,
clearly-labelled **synthetic** stand-in with the same shape, so the rest
of the pipeline has real (if small) data to run against today. Contains
`Document` and `QAExample` dataclasses used everywhere else in the
codebase, plus `get_synthetic_hallucinated_pair()`, which fabricates a
crude "hallucinated" counterpart to a correct answer — a stand-in for
what MedHallu provides natively, so `evaluate.py` has both a positive
and a negative example to score against right now.

### `retriever.py`
The **Medical Document Retrieval Module**. Embeds every document in the
corpus using a sentence-transformer model (`all-MiniLM-L6-v2` by
default) and retrieves the top-k most similar documents to a query via
cosine similarity. This is real, working dense retrieval — not a stub —
it will return genuinely relevant passages once pointed at a real
corpus. Also implements `reformulate_and_retrieve()`, used by the
pipeline when a claim needs re-retrieval: it widens the search by
appending the unsupported claim to the original query.

*Swap-in point:* once your corpus is large, replace the brute-force
NumPy similarity search with a FAISS/Chroma index — the public
`retrieve(query, top_k)` interface would not need to change.

### `generator.py`
The **Response Generation Module**. Defines an abstract `BaseGenerator`
interface with two implementations:
- `MockGenerator` — offline and deterministic. Builds an answer from the
  retrieved evidence and, with a configurable probability, appends one
  fabricated, ungrounded sentence — this is what makes the demo
  meaningful: there is always a real hallucination for the verifier to
  try to catch, without needing any LLM API key.
- `OpenAIGenerator` — a fully coded (not stubbed) wrapper around the
  OpenAI chat completions API, ready to use the moment you add an API
  key. Swap this for Anthropic/a local HF model by writing one more
  class that implements `BaseGenerator.generate()`.

`get_generator(backend="mock"|"openai")` is a factory so the rest of the
code never has to import a concrete class directly.

### `sentence_splitter.py`
The first half of the **Sentence-Level Verification Module**. Splits a
generated answer into individual sentences/claims, each of which gets
independently verified. Provides an NLTK-based splitter (used if NLTK is
installed) and a dependency-free regex-based fallback that protects
common medical abbreviations (`Dr.`, `mg.`, `e.g.`, etc.) from being
mistaken for sentence boundaries. `split_into_claims()` is the single
function everything else calls.

### `nli_verifier.py`
The second half of the **Sentence-Level Verification Module** — the
"NLI verifier" box in your architecture diagram. Wraps a Hugging Face
Natural Language Inference model (`facebook/bart-large-mnli` by
default) to score each (evidence, generated-sentence) pair for
entailment / neutral / contradiction probabilities. When multiple
evidence passages are available for one claim, it keeps whichever gives
the *highest* entailment score as the "best support" for that claim.
Returns a `VerificationResult` dataclass consumed by `abstention.py`.

### `abstention.py`
The **Tiered Abstention Module** — the core novelty of your project.
Pure decision logic, no ML calls: given a `VerificationResult` and the
thresholds in `config.py`, decides one of four actions:
- **DISPLAY** — entailment probability is high enough to trust the claim.
- **SUPPRESS** — contradiction probability is high; drop the sentence
  entirely.
- **RE_RETRIEVE** — evidence is too weak/inconclusive; ask for another
  retrieval attempt with a reformulated query.
- **FLAG** — still inconclusive after re-retrieval attempts; show the
  sentence but mark it `[⚠ UNVERIFIED]` rather than silently hiding
  potentially useful information.

`format_for_display()` turns a decision into the actual text the user
sees (or an empty string, for suppressed sentences).

### `pipeline.py`
The **orchestrator** — implements the full flowchart on slide 12 end to
end: retrieve → generate → split → (verify → decide, looping through
re-retrieval attempts up to `config.MAX_RE_RETRIEVAL_ATTEMPTS`) →
assemble the final response. Returns a `PipelineResult` containing both
the raw (unverified) LLM output and the final, verified answer, plus a
full per-sentence audit trail (`SentenceTrace`) so you can show/debug
exactly why each sentence was accepted, dropped, or flagged — useful for
your project demo/defense.

### `evaluate.py`
The **Evaluation Module**. Runs the pipeline across a set of QA examples
and computes the four metrics named in your Objectives slide:
**hallucination rate, precision, recall, and false abstention rate**
(each defined with its formula in the file's docstring). Currently runs
against a small synthetic evaluation set built by pairing each synthetic
QA example with a fabricated "hallucinated twin" — once you load
MedHallu/RAGTruth/MIRAGE/RAGCare-QA via `data_loader.py`, this file's
metric-computation logic does not need to change, only the source of
`examples` passed into `evaluate_on_examples()`.

### `main.py`
Thin CLI demo runner. Three modes:
```bash
python main.py                                  # runs 4 sample medical queries
python main.py --query "What treats hypertension?"  # ask one custom question
python main.py --evaluate                        # runs the Evaluation Module
```

### `requirements.txt`
All Python packages needed to actually run the pipeline (PyTorch,
Transformers, Sentence-Transformers, NLTK). `openai` is only needed if
you use the `OpenAIGenerator` backend.

---

## Running it

```bash
pip install -r requirements.txt
python main.py
```

This runs entirely offline against the built-in synthetic corpus and
`MockGenerator` — no API keys or datasets required to see the whole
DISPLAY / SUPPRESS / RE_RETRIEVE / FLAG pipeline work end to end.

Each module also has its own smoke test at the bottom (`if __name__ ==
"__main__":`), runnable individually, e.g. `python retriever.py` or
`python nli_verifier.py`.

## Wiring in your real datasets (the other 40%)

1. Download MedHallu, MIRAGE/MedRAG, RAGCare-QA, and raw PubMedQA (links
   are on your slide 15).
2. Set the four path constants in `config.py` (`MEDHALLU_PATH`,
   `MIRAGE_PATH`, `RAGCARE_QA_PATH`, `PUBMEDQA_PATH`).
3. Fill in the two `# TODO: real parsing logic here` blocks in
   `data_loader.py` (`load_retrieval_corpus`, `load_evaluation_examples`)
   — the expected schema for each dataset is documented in the
   surrounding comments.
4. Optionally swap `MockGenerator` for `OpenAIGenerator` (or your own
   `BaseGenerator` subclass) in `main.py` / `pipeline.py` once you decide
   which LLM backbone the project will use.
5. Re-tune the three thresholds in `config.py` using a validation split
   once real labelled data is available.

No other file needs to change — `retriever.py`, `nli_verifier.py`,
`abstention.py`, `pipeline.py`, and `evaluate.py` all operate on the
common `Document` / `QAExample` schema regardless of where the data came
from.
