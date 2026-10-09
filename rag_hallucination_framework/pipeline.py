"""
pipeline.py
-----------
Orchestrates the full flow shown in slide 12 (System Architecture) and
described step-by-step on slide 16 (Prototype / Research Framework):

    User query
      -> Retriever (retriever.py)
      -> Answer generator / LLM (generator.py)
      -> Sentence splitter (sentence_splitter.py)
      -> NLI verifier, per sentence (nli_verifier.py)
      -> Tiered abstention decision, per sentence (abstention.py)
           - RE_RETRIEVE loops back into the retriever with a
             reformulated query, up to config.MAX_RE_RETRIEVAL_ATTEMPTS
             times, before falling back to FLAG.
      -> Assembled final response (only DISPLAY/FLAG sentences are kept;
         SUPPRESS sentences are dropped)

This is the one file that "wires everything together"; every other
module can be developed, imported, and unit-tested independently of it.
"""

from dataclasses import dataclass, field
from typing import List

import config
from retriever import Retriever
from generator import BaseGenerator, get_generator
from sentence_splitter import split_into_claims
from nli_verifier import NLIVerifier, VerificationResult
from abstention import decide_action, format_for_display, AbstentionAction, AbstentionDecision


@dataclass
class SentenceTrace:
    """
    Full audit trail for one generated sentence, useful for the
    Evaluation Module (evaluate.py) and for debugging/demoing the system
    -- lets you show a professor/reviewer exactly why a sentence was
    displayed, suppressed, flagged, or re-retrieved.
    """
    original_sentence: str
    final_action: AbstentionAction
    final_decision: AbstentionDecision
    num_re_retrievals: int = 0
    displayed_text: str = ""


@dataclass
class PipelineResult:
    """Everything produced for a single user query, in one place."""
    query: str
    raw_answer: str
    final_response: str
    sentence_traces: List[SentenceTrace] = field(default_factory=list)


class HallucinationMitigationPipeline:
    """
    The main entry point described across slides 11, 12, and 16. Instantiate
    once (so the retriever's index and the NLI/generator models are loaded
    only once) and call `.run(query)` per user question.
    """

    def __init__(
        self,
        retriever: Retriever = None,
        generator: BaseGenerator = None,
        verifier: NLIVerifier = None,
    ):
        self.retriever = retriever or Retriever()
        self.generator = generator or get_generator("mock")
        self.verifier = verifier or NLIVerifier()

    def run(self, query: str) -> PipelineResult:
        """
        Executes the full DISPLAY/SUPPRESS/RE_RETRIEVE/FLAG pipeline for a
        single query and returns a PipelineResult with the final assembled
        answer plus a full per-sentence audit trail.
        """
        # --- Step 1: Retrieve evidence for the original query ---------
        retrieved = self.retriever.retrieve(query)
        evidence_docs = [doc for doc, _score in retrieved]

        # --- Step 2: Generate a draft answer grounded in that evidence -
        raw_answer = self.generator.generate(query, evidence_docs)

        # --- Step 3: Split the draft answer into individual claims -----
        claims = split_into_claims(raw_answer)

        # --- Step 4 & 5: Verify each claim, apply tiered abstention ----
        traces: List[SentenceTrace] = []
        for claim in claims:
            trace = self._verify_and_decide(
                query=query, claim=claim, initial_evidence_docs=evidence_docs
            )
            traces.append(trace)

        # --- Step 6: Assemble the final response ------------------------
        displayed_parts = [t.displayed_text for t in traces if t.displayed_text]
        final_response = " ".join(displayed_parts)

        return PipelineResult(
            query=query,
            raw_answer=raw_answer,
            final_response=final_response,
            sentence_traces=traces,
        )

    def _verify_and_decide(
        self, query: str, claim: str, initial_evidence_docs
    ) -> SentenceTrace:
        """
        Verifies a single claim and applies the abstention policy,
        looping through re-retrieval attempts as needed. This is where
        the RE_RETRIEVE branch of slide 12's flowchart is actually
        resolved into a final DISPLAY / SUPPRESS / FLAG outcome.
        """
        evidence_docs = initial_evidence_docs
        num_re_retrievals = 0

        while True:
            verification: VerificationResult = self.verifier.verify(claim, evidence_docs)
            decision = decide_action(verification)

            if decision.action != AbstentionAction.RE_RETRIEVE:
                break  # DISPLAY, SUPPRESS, or FLAG -- we're done

            if num_re_retrievals >= config.MAX_RE_RETRIEVAL_ATTEMPTS:
                # Exhausted our re-retrieval budget; fall back to FLAG
                # rather than looping forever.
                decision = AbstentionDecision(
                    action=AbstentionAction.FLAG,
                    reason=(
                        f"Re-retrieval attempted {num_re_retrievals} time(s) without "
                        "finding sufficient evidence; falling back to FLAG."
                    ),
                    verification=verification,
                )
                break

            # Attempt re-retrieval with a reformulated (widened) query.
            num_re_retrievals += 1
            reformulated_results = self.retriever.reformulate_and_retrieve(
                original_query=query, unsupported_claim=claim
            )
            evidence_docs = [doc for doc, _score in reformulated_results]

        displayed_text = format_for_display(decision)
        return SentenceTrace(
            original_sentence=claim,
            final_action=decision.action,
            final_decision=decision,
            num_re_retrievals=num_re_retrievals,
            displayed_text=displayed_text,
        )


if __name__ == "__main__":
    # Smoke test: `python pipeline.py`
    # Requires: pip install sentence-transformers transformers torch
    pipeline = HallucinationMitigationPipeline()
    result = pipeline.run("What is the first-line treatment for type 2 diabetes?")

    print("QUERY:", result.query)
    print("\nRAW (unverified) ANSWER:\n", result.raw_answer)
    print("\nPER-SENTENCE DECISIONS:")
    for trace in result.sentence_traces:
        print(f"  [{trace.final_action.value}] {trace.original_sentence}")
        print(f"      reason: {trace.final_decision.reason}")
    print("\nFINAL (verified) RESPONSE SHOWN TO USER:\n", result.final_response)
