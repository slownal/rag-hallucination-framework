"""
nli_verifier.py
----------------
Implements the "NLI verifier -> Checks each claim against source" box
from slide 12 (System Architecture), and the second half of the
"Sentence-Level Verification Module" from slide 11.

Core idea (standard textual entailment / NLI):
    Given a PREMISE (a retrieved evidence passage) and a HYPOTHESIS (one
    generated sentence/claim), an NLI model outputs probabilities for
    three labels:
        - entailment:    the premise supports the hypothesis
        - neutral:       the premise neither supports nor contradicts it
        - contradiction: the premise contradicts the hypothesis

A claim can have MULTIPLE retrieved evidence passages. We compute
entailment/contradiction against every passage and keep the passage with
the HIGHEST entailment probability as the "best support" for that claim,
following the same "verifies each generated sentence" language on slide 3
(Abstract) and 4 (Introduction).

This module is fully implemented and will genuinely run an NLI model
(default: facebook/bart-large-mnli) via Hugging Face `transformers` --
it is not a stub. The only thing missing until you have your dataset is
which evidence corpus/documents get passed in, which is handled by
retriever.py.
"""

from dataclasses import dataclass
from typing import List, Tuple

import config
from data_loader import Document


@dataclass
class VerificationResult:
    """
    Holds the outcome of verifying one generated sentence against the best
    matching evidence passage.
    """
    sentence: str
    best_evidence: Document
    entailment_prob: float
    neutral_prob: float
    contradiction_prob: float


class NLIVerifier:
    """
    Wraps a Hugging Face NLI (natural language inference) pipeline and
    exposes a `verify(sentence, evidence_docs)` method used by
    pipeline.py.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.NLI_MODEL_NAME
        self._pipeline = None  # loaded lazily, see _ensure_model_loaded()

        # bart-large-mnli's label order from the HF pipeline output is
        # typically ["contradiction", "neutral", "entailment"], but we
        # look labels up by NAME (not position) to stay robust to model
        # changes -- see `_extract_label_probs`.

    def _ensure_model_loaded(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as e:
                raise ImportError(
                    "transformers is required for the NLIVerifier. "
                    "Install with: pip install transformers torch"
                ) from e

            # "text-classification" with `top_k=None` returns scores for
            # every label; this is the flow HF recommends for NLI models
            # fine-tuned on 3-way entailment (MNLI-style) labels.
            self._pipeline = pipeline(
                task="text-classification",
                model=self.model_name,
                top_k=None,
                device=-1 if config.DEVICE == "cpu" else 0,
            )

    @staticmethod
    def _extract_label_probs(raw_output) -> Tuple[float, float, float]:
        """
        Normalises the pipeline's raw output (a list of
        {"label": ..., "score": ...} dicts, in arbitrary order) into
        (entailment_prob, neutral_prob, contradiction_prob).
        """
        probs = {item["label"].lower(): item["score"] for item in raw_output}
        entailment = probs.get("entailment", 0.0)
        neutral = probs.get("neutral", 0.0)
        contradiction = probs.get("contradiction", 0.0)
        return entailment, neutral, contradiction

    def _score_pair(self, premise: str, hypothesis: str) -> Tuple[float, float, float]:
        """
        Runs the NLI model on a single (premise, hypothesis) pair and
        returns (entailment_prob, neutral_prob, contradiction_prob).
        """
        self._ensure_model_loaded()
        # Most HF NLI pipelines accept a dict with "text" (premise) and
        # "text_pair" (hypothesis) fields via the underlying tokenizer.
        result = self._pipeline({"text": premise, "text_pair": hypothesis})
        # top_k=None makes the pipeline return a nested list (one list per
        # input); since we pass a single example, unwrap the outer list.
        if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
            result = result[0]
        return self._extract_label_probs(result)

    def verify(
        self, sentence: str, evidence_docs: List[Document]
    ) -> VerificationResult:
        """
        Verifies a single generated sentence against a list of candidate
        evidence documents (typically the retriever's top_k results).

        Returns a VerificationResult built from whichever evidence
        document yields the HIGHEST entailment probability -- i.e. the
        strongest available support for the claim.
        """
        if not evidence_docs:
            # No evidence at all -- treat as fully unsupported.
            return VerificationResult(
                sentence=sentence,
                best_evidence=None,
                entailment_prob=0.0,
                neutral_prob=0.0,
                contradiction_prob=0.0,
            )

        best_result = None
        best_entailment = -1.0

        for doc in evidence_docs:
            entailment, neutral, contradiction = self._score_pair(
                premise=doc.text, hypothesis=sentence
            )
            if entailment > best_entailment:
                best_entailment = entailment
                best_result = VerificationResult(
                    sentence=sentence,
                    best_evidence=doc,
                    entailment_prob=entailment,
                    neutral_prob=neutral,
                    contradiction_prob=contradiction,
                )

        return best_result


if __name__ == "__main__":
    # Smoke test: `python nli_verifier.py`
    # Requires `pip install transformers torch` to actually run.
    from data_loader import load_retrieval_corpus

    corpus = load_retrieval_corpus()
    verifier = NLIVerifier()

    supported_claim = "Metformin is used to treat type 2 diabetes."
    result = verifier.verify(supported_claim, corpus)
    print(f"Claim: {supported_claim}")
    print(f"  entailment={result.entailment_prob:.3f} "
          f"neutral={result.neutral_prob:.3f} "
          f"contradiction={result.contradiction_prob:.3f}")

    fabricated_claim = "This medication cures diabetes within 24 hours with no side effects."
    result = verifier.verify(fabricated_claim, corpus)
    print(f"Claim: {fabricated_claim}")
    print(f"  entailment={result.entailment_prob:.3f} "
          f"neutral={result.neutral_prob:.3f} "
          f"contradiction={result.contradiction_prob:.3f}")
