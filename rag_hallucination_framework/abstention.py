"""
abstention.py
--------------
Implements the "Tiered Abstention Module" from slide 11 and the
"Confidence check -> Entailed or not -> Accept sentence / Abstention
action" branch of slide 12 (System Architecture). This is the central
novelty of the project described in the Abstract (slide 3): rather than a
binary accept/reject, each sentence is routed to one of FOUR actions.

The four tiers, matching slide 9 (Research Gap) and slide 11:
    DISPLAY      - sentence is well supported by retrieved evidence -> show
                   it to the user as-is.
    SUPPRESS     - sentence is contradicted by retrieved evidence -> hide
                   it entirely (never show a claim the evidence disputes).
    RE_RETRIEVE  - evidence is insufficient/inconclusive -> try retrieving
                   again with a reformulated query before giving up.
    FLAG         - after re-retrieval still inconclusive (or borderline
                   neutral) -> show the sentence but mark it as
                   "unverified" so the user knows to treat it with
                   caution, instead of silently hiding useful information.

This module is pure decision logic -- it takes a VerificationResult (from
nli_verifier.py) and the thresholds in config.py, and returns an
AbstentionDecision. It does not itself call the retriever; pipeline.py
is responsible for actually invoking re-retrieval when this module
requests it, since only the pipeline holds the retriever + original
query needed to do so.
"""

from dataclasses import dataclass
from enum import Enum

import config
from nli_verifier import VerificationResult


class AbstentionAction(str, Enum):
    DISPLAY = "DISPLAY"
    SUPPRESS = "SUPPRESS"
    RE_RETRIEVE = "RE_RETRIEVE"
    FLAG = "FLAG"


@dataclass
class AbstentionDecision:
    action: AbstentionAction
    reason: str
    verification: VerificationResult


def decide_action(verification: VerificationResult) -> AbstentionDecision:
    """
    Applies the tiered abstention policy (config.py thresholds) to a
    single sentence's verification result.

    Decision order matters -- contradiction is checked before "insufficient
    evidence" so that a confidently-contradicted claim is SUPPRESSed
    rather than sent back for pointless re-retrieval.
    """
    e = verification.entailment_prob
    c = verification.contradiction_prob

    if e >= config.ENTAILMENT_ACCEPT_THRESHOLD:
        return AbstentionDecision(
            action=AbstentionAction.DISPLAY,
            reason=(
                f"Entailment probability {e:.3f} >= accept threshold "
                f"{config.ENTAILMENT_ACCEPT_THRESHOLD}; evidence supports the claim."
            ),
            verification=verification,
        )

    if c >= config.CONTRADICTION_SUPPRESS_THRESHOLD:
        return AbstentionDecision(
            action=AbstentionAction.SUPPRESS,
            reason=(
                f"Contradiction probability {c:.3f} >= suppress threshold "
                f"{config.CONTRADICTION_SUPPRESS_THRESHOLD}; evidence disputes the claim."
            ),
            verification=verification,
        )

    if e < config.RETRIEVAL_INSUFFICIENT_THRESHOLD:
        return AbstentionDecision(
            action=AbstentionAction.RE_RETRIEVE,
            reason=(
                f"Entailment probability {e:.3f} < insufficient-evidence threshold "
                f"{config.RETRIEVAL_INSUFFICIENT_THRESHOLD}; retrieved evidence is too "
                "weak to judge -- attempt re-retrieval."
            ),
            verification=verification,
        )

    # Borderline zone: not confidently entailed, not confidently
    # contradicted, and not so low that immediate re-retrieval is clearly
    # warranted. Flag it for the user rather than silently guessing.
    return AbstentionDecision(
        action=AbstentionAction.FLAG,
        reason=(
            f"Entailment ({e:.3f}) and contradiction ({verification.contradiction_prob:.3f}) "
            "are both inconclusive; flagging sentence as unverified."
        ),
        verification=verification,
    )


def format_for_display(decision: AbstentionDecision) -> str:
    """
    Turns an AbstentionDecision into the text actually shown to the end
    user, implementing the "Assembled response -> Delivered to user
    interface" step from slide 12. SUPPRESSed sentences return an empty
    string (i.e. they are dropped from the final answer entirely).
    """
    sentence = decision.verification.sentence

    if decision.action == AbstentionAction.DISPLAY:
        return sentence
    elif decision.action == AbstentionAction.SUPPRESS:
        return ""  # dropped from the final answer
    elif decision.action == AbstentionAction.FLAG:
        return f"{sentence} [⚠ UNVERIFIED: not confidently supported by retrieved evidence]"
    elif decision.action == AbstentionAction.RE_RETRIEVE:
        # By the time format_for_display is called, RE_RETRIEVE should
        # already have been resolved into one of the other three actions
        # by pipeline.py (see MAX_RE_RETRIEVAL_ATTEMPTS). This branch is a
        # safety net in case re-retrieval was exhausted without a final
        # decision being recorded.
        return f"{sentence} [⚠ UNVERIFIED: evidence remained insufficient after re-retrieval]"
    else:
        raise ValueError(f"Unknown abstention action: {decision.action}")
