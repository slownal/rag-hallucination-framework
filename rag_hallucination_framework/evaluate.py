"""
evaluate.py
-----------
Implements the "Evaluation Module" from slide 11 and the evaluation goal
from slide 10 (Objectives): "Evaluate the framework using MIRAGE,
RAGCare-QA, and RAGTruth with metrics such as hallucination rate,
precision, recall, and false abstention rate."

Metric definitions:
  hallucination_rate: fraction of all sentences that are ground-truth
                      hallucinations BEFORE abstention is applied.
  precision         : of all SUPPRESS/FLAG decisions, what fraction were
                      truly hallucinated?  TP / (TP + FP)
  recall            : of all truly hallucinated sentences, what fraction
                      did the system catch?  TP / (TP + FN)
  false_abstention_rate (FAR): of all truly correct sentences, what
                      fraction did the system incorrectly suppress/flag?
                      FP / (FP + TN)

Supports all three datasets:
  - medhallu  : provides both correct and hallucinated answers natively
  - pubmedqa  : only correct answers (no hallucinated pairs natively)
  - mirage    : multiple-choice QA (correct answers only)
  - synthetic : built-in small dataset for offline development
"""

from dataclasses import dataclass
from typing import List

from abstention import AbstentionAction
from data_loader import (
    QAExample,
    load_evaluation_examples,
    get_synthetic_hallucinated_pair,
)
from pipeline import HallucinationMitigationPipeline


@dataclass
class EvaluationMetrics:
    hallucination_rate: float
    precision: float
    recall: float
    false_abstention_rate: float
    num_sentences_evaluated: int
    num_true_hallucinations: int
    num_flagged_or_suppressed: int
    dataset_name: str = "unknown"


def _is_flagged_by_system(action: AbstentionAction) -> bool:
    """SUPPRESS and FLAG both count as 'system pushed back on this claim'."""
    return action in (AbstentionAction.SUPPRESS, AbstentionAction.FLAG)


def evaluate_on_examples(
    pipeline: HallucinationMitigationPipeline,
    examples: List[QAExample],
    dataset_name: str = "unknown",
) -> EvaluationMetrics:
    """
    Runs the pipeline on each QAExample's question, then compares the
    system's per-sentence abstention decisions against each example's
    ground-truth `is_hallucinated` label.

    For MedHallu: each row is paired as (correct, hallucinated), giving
    sentence-level precision/recall/FAR.

    For PubMedQA / MIRAGE: all examples have is_hallucinated=False, so
    hallucination_rate=0 and recall=undefined; FAR and precision are still
    informative.
    """
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0
    total_sentences = 0
    total_hallucinated = 0

    for example in examples:
        result = pipeline.run(example.question)
        ground_truth_hallucinated = bool(example.is_hallucinated)

        for trace in result.sentence_traces:
            total_sentences += 1
            system_flagged = _is_flagged_by_system(trace.final_action)

            if ground_truth_hallucinated:
                total_hallucinated += 1
                if system_flagged:
                    true_positive += 1
                else:
                    false_negative += 1
            else:
                if system_flagged:
                    false_positive += 1
                else:
                    true_negative += 1

    hallucination_rate = total_hallucinated / total_sentences if total_sentences else 0.0
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )
    false_abstention_rate = (
        false_positive / (false_positive + true_negative)
        if (false_positive + true_negative) > 0
        else 0.0
    )

    return EvaluationMetrics(
        hallucination_rate=hallucination_rate,
        precision=precision,
        recall=recall,
        false_abstention_rate=false_abstention_rate,
        num_sentences_evaluated=total_sentences,
        num_true_hallucinations=total_hallucinated,
        num_flagged_or_suppressed=true_positive + false_positive,
        dataset_name=dataset_name,
    )


def build_synthetic_eval_set() -> List[QAExample]:
    """
    Builds a small mixed evaluation set for offline testing:
    every synthetic example is paired with a hallucinated counterpart.
    This mirrors what MedHallu provides at much larger scale.
    """
    base_examples = load_evaluation_examples("synthetic")
    eval_set = list(base_examples)
    for ex in base_examples:
        eval_set.append(get_synthetic_hallucinated_pair(ex))
    return eval_set


def print_metrics(metrics: EvaluationMetrics):
    """Pretty-prints an EvaluationMetrics object."""
    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS  —  dataset: {metrics.dataset_name}")
    print(f"{'='*60}")
    print(f"  Sentences evaluated:        {metrics.num_sentences_evaluated}")
    print(f"  True hallucinations:        {metrics.num_true_hallucinations}")
    print(f"  Flagged/suppressed by sys:  {metrics.num_flagged_or_suppressed}")
    print(f"  Hallucination rate:         {metrics.hallucination_rate:.3f}")
    print(f"  Precision:                  {metrics.precision:.3f}")
    print(f"  Recall:                     {metrics.recall:.3f}")
    print(f"  False abstention rate:      {metrics.false_abstention_rate:.3f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Smoke test: `python evaluate.py`
    # Requires: pip install sentence-transformers transformers torch datasets
    pipeline = HallucinationMitigationPipeline()
    eval_examples = build_synthetic_eval_set()

    metrics = evaluate_on_examples(pipeline, eval_examples, dataset_name="synthetic")
    print_metrics(metrics)
