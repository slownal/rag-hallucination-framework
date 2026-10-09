"""
main.py
-------
Command-line entry point for the RAG Hallucination Mitigation Framework.

USAGE:
    # Run a demo with sample medical queries (uses synthetic corpus)
    python main.py

    # Ask a single custom question
    python main.py --query "What treats hypertension?"

    # Evaluate on a real dataset
    python main.py --evaluate --dataset medhallu
    python main.py --evaluate --dataset pubmedqa
    python main.py --evaluate --dataset mirage
    python main.py --evaluate --dataset synthetic

    # Use different generator backends
    python main.py --backend mock    # default: offline, no API key needed
    python main.py --backend openai  # requires OPENAI_API_KEY env variable

    # Limit dataset size (for quick testing)
    python main.py --evaluate --dataset medhallu --max-examples 50
"""

import argparse
import sys

from pipeline import HallucinationMitigationPipeline
from generator import get_generator
from retriever import Retriever
from nli_verifier import NLIVerifier
from evaluate import evaluate_on_examples, build_synthetic_eval_set, print_metrics
from data_loader import load_evaluation_examples, get_synthetic_hallucinated_pair


SAMPLE_QUERIES = [
    "What is the first-line treatment for type 2 diabetes?",
    "What are the side effects of metformin?",
    "How does aspirin help prevent heart attacks?",
    "What defines hypertension?",
]


def print_result(result):
    print("=" * 80)
    print(f"QUERY: {result.query}")
    print("-" * 80)
    print("RAW (unverified) GENERATED ANSWER:")
    print(f"  {result.raw_answer[:300]}{'...' if len(result.raw_answer) > 300 else ''}")
    print("-" * 80)
    print("PER-SENTENCE ABSTENTION DECISIONS:")
    for trace in result.sentence_traces:
        re_retry_note = (
            f" (after {trace.num_re_retrievals} re-retrieval attempt(s))"
            if trace.num_re_retrievals
            else ""
        )
        print(f"  [{trace.final_action.value}]{re_retry_note}")
        print(f"    Sentence : {trace.original_sentence[:120]}...")
        print(f"    Reason   : {trace.final_decision.reason[:100]}")
    print("-" * 80)
    print("FINAL RESPONSE SHOWN TO USER:")
    response = result.final_response if result.final_response else "(no sentences survived verification)"
    print(f"  {response[:400]}{'...' if len(response) > 400 else ''}")
    print("=" * 80 + "\n")


def run_demo(pipeline: HallucinationMitigationPipeline):
    print(
        "\nRunning the RAG hallucination-mitigation pipeline on sample "
        "medical questions against the built-in corpus...\n"
    )
    for query in SAMPLE_QUERIES:
        result = pipeline.run(query)
        print_result(result)


def run_single_query(pipeline: HallucinationMitigationPipeline, query: str):
    result = pipeline.run(query)
    print_result(result)


def run_evaluation(
    pipeline: HallucinationMitigationPipeline,
    dataset_name: str,
    max_examples: int = None,
):
    print(f"\nRunning Evaluation Module on dataset: '{dataset_name}'\n")

    if dataset_name == "synthetic":
        eval_examples = build_synthetic_eval_set()
    else:
        examples = load_evaluation_examples(dataset_name)
        if max_examples:
            examples = examples[:max_examples]

        # For datasets without native hallucinated pairs (pubmedqa, mirage),
        # augment with synthetic corruptions so precision/recall can be computed.
        if dataset_name in ("pubmedqa", "mirage"):
            print(
                f"  Note: '{dataset_name}' has no native hallucinated pairs. "
                "Augmenting with synthetic corruptions for precision/recall scoring."
            )
            eval_examples = list(examples)
            for ex in examples:
                eval_examples.append(get_synthetic_hallucinated_pair(ex))
        else:
            # MedHallu already contains both correct and hallucinated examples
            eval_examples = examples

    print(f"  Total examples to evaluate: {len(eval_examples)}\n")
    metrics = evaluate_on_examples(pipeline, eval_examples, dataset_name=dataset_name)
    print_metrics(metrics)


def build_pipeline(backend: str = "mock") -> HallucinationMitigationPipeline:
    """Constructs and returns the pipeline with the chosen generator backend."""
    print(f"\n[main] Initializing pipeline (generator backend: '{backend}') ...")
    generator = get_generator(backend)
    retriever = Retriever()
    verifier = NLIVerifier()
    return HallucinationMitigationPipeline(
        retriever=retriever,
        generator=generator,
        verifier=verifier,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sentence-Level Hallucination Detection and Tiered Abstention "
            "Framework — end-to-end demo runner."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Ask a single custom medical question.",
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run the Evaluation Module.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="medhallu",
        choices=["medhallu", "pubmedqa", "mirage", "synthetic"],
        help=(
            "Dataset to use for evaluation. "
            "'medhallu' (default) provides native hallucinated pairs; "
            "'pubmedqa'/'mirage' are augmented with synthetic corruptions."
        ),
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="mock",
        choices=["mock", "openai"],
        help=(
            "Generator backend. 'mock' (default) runs fully offline. "
            "'openai' requires OPENAI_API_KEY environment variable."
        ),
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit the number of evaluation examples (useful for quick runs).",
    )
    args = parser.parse_args()

    pipeline = build_pipeline(args.backend)

    if args.evaluate:
        run_evaluation(pipeline, args.dataset, max_examples=args.max_examples)
    elif args.query:
        run_single_query(pipeline, args.query)
    else:
        run_demo(pipeline)


if __name__ == "__main__":
    main()
