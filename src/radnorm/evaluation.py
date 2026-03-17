from __future__ import annotations

from collections import Counter
from typing import Any


ERROR_CLASS_TO_FAMILY = {
    "wrong_match_extremity": "extremity_underspecification",
    "wrong_match_chest": "chest_underspecification",
    "wrong_match_abdomen": "abdomen_underspecification",
    "wrong_match_pelvis": "pelvis_underspecification",
    "missed_match": "under_matching",
    "out_of_scope_miss": "out_of_scope_handling",
    "other": "other",
}


def classify_failure(gold: dict[str, Any], pred: dict[str, Any]) -> str:
    body_part = gold["expected_attributes"].get("body_part")
    if gold["requires_review"] and pred["decision"] == "match":
        if body_part in {"knee", "hand", "ankle"}:
            return "wrong_match_extremity"
        if body_part == "chest":
            return "wrong_match_chest"
        if body_part == "abdomen":
            return "wrong_match_abdomen"
        if body_part == "pelvis":
            return "wrong_match_pelvis"
    if (not gold["requires_review"]) and pred["decision"] == "needs_review":
        return "missed_match"
    if gold["requires_review"] and gold["expected_code"] is None and gold["source_text"].lower().startswith(("ct ", "mri ", "us ", "fluoro")):
        return "out_of_scope_miss"
    return "other"


def evaluate_predictions(predictions: list[dict[str, Any]], dataset: list[dict[str, Any]]) -> dict[str, Any]:
    if len(predictions) != len(dataset):
        raise ValueError("Prediction count does not match dataset size.")
    total = len(dataset)
    correct = 0
    failures: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()

    for pred, gold in zip(predictions, dataset):
        expected_decision = "needs_review" if gold["requires_review"] else "match"
        expected_id = gold["expected_code"]
        expected_output = {
            "decision": expected_decision,
            "canonical_id": expected_id,
            "reason": "expected_reference_output",
        }
        is_correct = pred["decision"] == expected_decision and pred.get("canonical_id") == expected_id
        if is_correct:
            correct += 1
            continue
        error_class = classify_failure(gold, pred)
        failure_counts[error_class] += 1
        failures.append(
            {
                "record_id": gold["record_id"],
                "hospital_id": gold["hospital_id"],
                "source_text": gold["source_text"],
                "body_part": gold["expected_attributes"].get("body_part"),
                "expected_output": expected_output,
                "predicted_output": {
                    "decision": pred["decision"],
                    "canonical_id": pred.get("canonical_id"),
                    "reason": pred.get("reason", ""),
                },
                "expected_attributes": gold["expected_attributes"],
                "error_class": error_class,
                "error_family": ERROR_CLASS_TO_FAMILY.get(error_class, "other"),
            }
        )

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "failures": failures,
        "failure_counts": dict(sorted(failure_counts.items())),
    }
