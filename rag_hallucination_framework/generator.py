"""
generator.py
------------
Implements the "Response Generation Module" and the
"Answer generator (LLM) -> Drafts answer using context" box from slide 12.

Design note: this module defines a small abstract interface
(`BaseGenerator`) plus two concrete implementations:

  1. `MockGenerator`     - a deterministic, offline, template-based
                            generator. It needs no API key and no
                            internet access, so the whole pipeline can be
                            run and demoed end-to-end right now. It is
                            deliberately written to sometimes produce an
                            UNSUPPORTED sentence, so that the downstream
                            NLI verifier / abstention modules have
                            something real to catch -- this is what makes
                            the demo meaningful rather than trivial.

  2. `OpenAIGenerator`    - a thin wrapper showing exactly where to plug in
                            a real LLM call once you decide on a
                            provider/model (OpenAI, Anthropic, a local HF
                            causal LM, etc.). It is fully implemented but
                            will only run once you supply an API key.

pipeline.py depends only on the `BaseGenerator` interface, so switching
between them (or adding a third backend) requires no changes elsewhere.
"""

from abc import ABC, abstractmethod
from typing import List
import random

import config
from data_loader import Document


class BaseGenerator(ABC):
    """Common interface every answer generator must implement."""

    @abstractmethod
    def generate(self, query: str, context_docs: List[Document]) -> str:
        """
        Args:
            query: the user's medical question.
            context_docs: the retrieved evidence passages to ground the
                          answer in.
        Returns:
            A generated answer as a single string (possibly multiple
            sentences -- sentence_splitter.py breaks it apart later).
        """
        raise NotImplementedError


class MockGenerator(BaseGenerator):
    """
    Offline, deterministic stand-in for a real LLM. Useful for:
      - running the full pipeline with zero external dependencies/cost,
      - unit-testing the verifier + abstention logic against KNOWN
        supported and unsupported sentences.

    Behaviour: builds one sentence per retrieved document (a supported
    claim, lightly paraphrased from the evidence) and then, with some
    probability, appends one fabricated sentence that is NOT grounded in
    any retrieved document -- simulating a hallucination for the
    verifier to catch.
    """

    def __init__(self, hallucination_probability: float = 0.5, seed: int = None):
        self.hallucination_probability = hallucination_probability
        self._rng = random.Random(seed if seed is not None else config.RANDOM_SEED)

        # A small bank of plausible-sounding but ungrounded medical
        # sentences used to simulate hallucinations in demo mode. In a
        # real LLM these would instead emerge naturally from the model's
        # own (sometimes incorrect) generations.
        self._fabricated_claims = [
            "This treatment is guaranteed to work within 24 hours for all patients.",
            "No side effects have ever been reported for this medication.",
            "This condition can be completely cured without any medical supervision.",
            "This drug is safe to combine with any other medication in any dose.",
        ]

    def generate(self, query: str, context_docs: List[Document]) -> str:
        sentences = []
        for doc in context_docs:
            # Simple, deterministic "paraphrase": take the evidence text
            # as-is. A real LLM would compress/rephrase it; we keep it
            # verbatim here so the entailment relationship is obvious and
            # the demo is easy to reason about.
            sentences.append(doc.text)

        if self._rng.random() < self.hallucination_probability:
            sentences.append(self._rng.choice(self._fabricated_claims))

        return " ".join(sentences)


class OpenAIGenerator(BaseGenerator):
    """
    Real LLM-backed generator. Requires the `openai` package and an
    OPENAI_API_KEY environment variable.

        pip install openai
        export OPENAI_API_KEY=sk-...

    TODO (once you have API access / a chosen provider):
      - Swap this out for whichever provider your project standardises on
        (this class can serve as a template either way -- the shape of
        `generate()` will not need to change for pipeline.py).
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.GENERATOR_MODEL_NAME
        try:
            import openai  # noqa: F401
            self._openai = openai
        except ImportError as e:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            ) from e

    def generate(self, query: str, context_docs: List[Document]) -> str:
        context_text = "\n\n".join(
            f"[Source {i+1}] {doc.text}" for i, doc in enumerate(context_docs)
        )
        system_prompt = (
            "You are a medical assistant. Answer the user's question using "
            "ONLY the information in the provided sources. Do not invent "
            "facts that are not supported by the sources."
        )
        user_prompt = f"Sources:\n{context_text}\n\nQuestion: {query}\nAnswer:"

        client = self._openai.OpenAI()  # picks up OPENAI_API_KEY from env
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content


def get_generator(backend: str = "mock", **kwargs) -> BaseGenerator:
    """
    Factory function so pipeline.py / main.py can select a backend by
    name (e.g. from a CLI flag or config value) without importing the
    concrete classes directly.
    """
    if backend == "mock":
        return MockGenerator(**kwargs)
    elif backend == "openai":
        return OpenAIGenerator(**kwargs)
    else:
        raise ValueError(f"Unknown generator backend: {backend}")


if __name__ == "__main__":
    # Smoke test: `python generator.py`
    from data_loader import load_retrieval_corpus

    docs = load_retrieval_corpus()[:2]
    gen = MockGenerator(hallucination_probability=1.0)  # force a hallucination
    answer = gen.generate("What treats type 2 diabetes?", docs)
    print("Generated answer:\n", answer)
