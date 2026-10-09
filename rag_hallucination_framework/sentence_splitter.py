"""
sentence_splitter.py
---------------------
Implements the "Sentence splitter -> Decomposes answer into claims" box
from slide 12 (System Architecture), which is the first half of the
"Sentence-Level Verification Module" from slide 11.

This module deliberately has NO dependency on the retrieved evidence or
the NLI model -- it is a pure text-processing step, so it can be unit
tested trivially and reused regardless of which generator or verifier is
plugged in.

Two implementations are provided:
  - `split_with_regex`: a dependency-free fallback using a regex-based
    sentence boundary heuristic. Good enough for demo purposes and for
    environments where installing NLTK/spaCy is inconvenient.
  - `split_with_nltk`: a more linguistically robust splitter using NLTK's
    punkt tokenizer, used automatically when NLTK is available.

`split_into_claims()` is the public function pipeline.py actually calls;
it picks whichever backend is available and normalises the output.
"""

import re
from typing import List


def split_with_regex(text: str) -> List[str]:
    """
    Dependency-free sentence splitter. Splits on '.', '!', '?' followed by
    whitespace and a capital letter, while trying to avoid breaking on
    common abbreviations (Dr., e.g., i.e., mg., etc.).

    This is intentionally simple -- it will not handle every edge case in
    medical text (e.g. "Dr. Smith prescribed 5 mg. twice daily.") perfectly,
    but works well enough for prototyping. Swap in `split_with_nltk` (or a
    clinical-text-specific tokenizer such as scispaCy) for higher accuracy.
    """
    # Protect common abbreviations from being treated as sentence boundaries
    protected = text
    abbreviations = ["Dr.", "Mr.", "Mrs.", "e.g.", "i.e.", "vs.", "etc.", "mg.", "mL."]
    placeholder_map = {}
    for i, abbr in enumerate(abbreviations):
        placeholder = f"__ABBR{i}__"
        placeholder_map[placeholder] = abbr
        protected = protected.replace(abbr, placeholder)

    # Split on sentence-ending punctuation followed by whitespace
    raw_sentences = re.split(r"(?<=[.!?])\s+", protected)

    # Restore protected abbreviations
    sentences = []
    for sent in raw_sentences:
        for placeholder, abbr in placeholder_map.items():
            sent = sent.replace(placeholder, abbr)
        sent = sent.strip()
        if sent:
            sentences.append(sent)
    return sentences


def split_with_nltk(text: str) -> List[str]:
    """Uses NLTK's punkt sentence tokenizer, if installed."""
    import nltk

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    from nltk.tokenize import sent_tokenize

    return [s.strip() for s in sent_tokenize(text) if s.strip()]


def split_into_claims(text: str, prefer_nltk: bool = True) -> List[str]:
    """
    Public entry point used by pipeline.py. Decomposes a generated answer
    into individual sentence-level claims, each of which will be
    independently verified by nli_verifier.py.

    Args:
        text: the full generated answer string.
        prefer_nltk: if True (default), tries NLTK first and silently
                     falls back to the regex splitter if NLTK is not
                     installed.
    """
    if not text or not text.strip():
        return []

    if prefer_nltk:
        try:
            return split_with_nltk(text)
        except ImportError:
            pass  # fall through to regex splitter

    return split_with_regex(text)


if __name__ == "__main__":
    # Smoke test: `python sentence_splitter.py`
    sample = (
        "Metformin is the first-line treatment for type 2 diabetes. "
        "Dr. Smith recommends starting at a low dose, e.g. 500 mg. "
        "Common side effects include nausea and diarrhea!"
    )
    for i, sent in enumerate(split_into_claims(sample), start=1):
        print(f"{i}. {sent}")
