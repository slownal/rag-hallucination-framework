"""
retriever.py
------------
Implements the "Medical Document Retrieval Module" from slide 11
(Module Identification) and the "Retriever -> Fetches relevant chunks"
box from slide 12 (System Architecture).

Approach: dense retrieval using sentence embeddings + cosine similarity.
This is a legitimate, working retrieval method (not a stub) -- it will
retrieve real nearest-neighbour passages from whatever corpus it is given,
including the synthetic corpus from data_loader.py today, and a real
medical corpus (PubMedQA / MIRAGE) once you plug that in.

Swap-in points for later:
  - Replace the brute-force cosine-similarity search with a proper vector
    index (e.g. FAISS, Chroma, Qdrant) once the corpus is large -- the
    public interface (retrieve(query, top_k)) would stay identical.
  - Add BM25 as a second signal and combine scores for hybrid search, as
    mentioned in slide 13 ("Retrieve relevant medical documents using
    vector/hybrid search").
"""

from typing import List, Tuple
import numpy as np

import config
from data_loader import Document, load_retrieval_corpus


class Retriever:
    """
    Wraps a sentence-embedding model and an in-memory list of Documents,
    exposing a simple `retrieve(query, top_k)` method used by pipeline.py.
    """

    def __init__(self, documents: List[Document] = None, model_name: str = None):
        """
        Args:
            documents: list of Document objects to index. If None, loads
                       the corpus via data_loader.load_retrieval_corpus().
            model_name: sentence-transformers model id. Defaults to
                        config.RETRIEVER_EMBEDDING_MODEL.
        """
        self.model_name = model_name or config.RETRIEVER_EMBEDDING_MODEL
        self.documents: List[Document] = documents or load_retrieval_corpus()

        # The embedding model is loaded lazily so that importing this module
        # (e.g. for unit tests that don't need real embeddings) doesn't
        # force a slow download every time.
        self._model = None
        self._doc_embeddings = None

    # ----------------------------------------------------------------
    # Model / index setup
    # ----------------------------------------------------------------
    def _ensure_model_loaded(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for the Retriever. "
                    "Install with: pip install sentence-transformers"
                ) from e
            self._model = SentenceTransformer(self.model_name, device=config.DEVICE)

    def build_index(self):
        """
        Embeds every document in self.documents and caches the resulting
        matrix in self._doc_embeddings. Must be called once before
        `retrieve()` (pipeline.py calls this automatically).

        In a production system with a large corpus, this step would be
        replaced by building/loading a persistent FAISS/Chroma index
        instead of holding everything in a numpy array in memory.
        """
        self._ensure_model_loaded()
        texts = [doc.text for doc in self.documents]
        embeddings = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        self._doc_embeddings = embeddings  # shape: (num_docs, embedding_dim)

    # ----------------------------------------------------------------
    # Retrieval
    # ----------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[Document, float]]:
        """
        Returns the top_k documents most similar to `query`, each paired
        with its cosine-similarity score (higher = more relevant).

        Args:
            query: the user's medical question (or a reformulated query
                   during re-retrieval -- see abstention.py).
            top_k: number of documents to return. Defaults to
                   config.TOP_K_RETRIEVAL.
        """
        if self._doc_embeddings is None:
            self.build_index()

        top_k = top_k or config.TOP_K_RETRIEVAL
        self._ensure_model_loaded()

        query_embedding = self._model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]

        # Cosine similarity is a simple dot product because embeddings
        # were normalised (normalize_embeddings=True) above.
        scores = self._doc_embeddings @ query_embedding  # shape: (num_docs,)

        # argsort descending, take top_k
        top_indices = np.argsort(-scores)[:top_k]
        return [(self.documents[i], float(scores[i])) for i in top_indices]

    def reformulate_and_retrieve(
        self, original_query: str, unsupported_claim: str, top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        Implements the "re-retrieval" branch of the tiered abstention
        policy (slide 11: "Tiered Abstention Module ... Re-retrieved").

        When a generated sentence cannot be verified against the initially
        retrieved evidence, we widen the search by appending the
        unsupported claim itself to the query -- a simple but effective
        query-reformulation heuristic. A more advanced version could use
        an LLM to rewrite the query, or expand it with medical synonyms /
        UMLS concept expansion.
        """
        reformulated_query = f"{original_query} {unsupported_claim}"
        return self.retrieve(reformulated_query, top_k=top_k)


if __name__ == "__main__":
    # Smoke test: `python retriever.py`
    # Requires `pip install sentence-transformers` to actually run; the
    # rest of the codebase does not require this to be run at import time.
    retriever = Retriever()
    results = retriever.retrieve("What treats type 2 diabetes?")
    for doc, score in results:
        print(f"[{score:.3f}] {doc.doc_id}: {doc.text[:80]}...")
