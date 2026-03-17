from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .parser import parse_order


def build_benchmark_profile(dataset: list[dict[str, Any]]) -> dict[str, Any]:
    body_counts: Counter[str] = Counter()
    review_counts: Counter[str] = Counter()
    attr_presence: Counter[str] = Counter()
    hospital_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()

    for row in dataset:
        attrs = row["expected_attributes"]
        body_counts[attrs.get("body_part") or "unknown"] += 1
        review_counts["needs_review" if row["requires_review"] else "match"] += 1
        hospital_counts[row["hospital_id"]] += 1

        parsed = parse_order(row["source_text"])
        if any(ord(ch) > 127 for ch in row["source_text"]):
            language_counts["non_ascii"] += 1
        elif any(tok in row["source_text"].lower() for tok in [" radiografia", " vista", " proyecciones", " torax", " mano", " tobillo", " rodilla"]):
            language_counts["spanish_like"] += 1
        else:
            language_counts["english_like"] += 1

        if parsed["laterality"] not in {"unknown", "none"}:
            attr_presence["explicit_laterality"] += 1
        if parsed["view_count"] is not None or parsed["view_pattern"] is not None:
            attr_presence["explicit_view_information"] += 1
        if parsed["portable"] is True:
            attr_presence["explicit_portable"] += 1
        if parsed["projection"] is not None:
            attr_presence["explicit_projection"] += 1
        if parsed["subtype"] == "KUB":
            attr_presence["explicit_kub"] += 1
        if parsed["modality"] and parsed["modality"] != "XR":
            attr_presence["non_xr_orders"] += 1

    return {
        "dataset_size": len(dataset),
        "decision_balance": dict(review_counts),
        "body_part_distribution": dict(body_counts),
        "attribute_presence": dict(attr_presence),
        "hospital_distribution": dict(hospital_counts),
        "language_mix": dict(language_counts),
    }


def build_optimizer_payload(
    profile: dict[str, Any],
    evaluation: dict[str, Any],
    analyses: list[dict[str, Any]],
    max_failures_for_optimizer: int,
) -> dict[str, Any]:
    failure_by_body: dict[str, int] = defaultdict(int)
    family_counts: Counter[str] = Counter()
    analyzer_category_counts: Counter[str] = Counter()

    for failure, analysis in zip(evaluation["failures"], analyses):
        if failure["body_part"] is not None:
            failure_by_body[failure["body_part"]] += 1
        family_counts[failure["error_family"]] += 1
        analyzer_category_counts[analysis["failure_category"]] += 1

    failure_cases = []
    for failure, analysis in list(zip(evaluation["failures"], analyses))[:max_failures_for_optimizer]:
        failure_cases.append(
            {
                "record_id": failure["record_id"],
                "normalizer_user_input": {
                    "order_text": failure["source_text"],
                },
                "model_output": failure["predicted_output"],
                "expected_output": failure["expected_output"],
                "analysis": analysis,
            }
        )

    return {
        "dataset_profile": profile,
        "current_metrics": {
            "accuracy": evaluation["accuracy"],
            "correct": evaluation["correct"],
            "total": evaluation["total"],
            "failure_counts": evaluation["failure_counts"],
            "failure_family_counts": dict(family_counts),
            "failure_body_part_counts": dict(failure_by_body),
            "analyzer_category_counts": dict(analyzer_category_counts),
        },
        "analyzed_failures": failure_cases,
    }
