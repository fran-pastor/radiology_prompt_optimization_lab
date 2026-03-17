from __future__ import annotations

from typing import Any

from ..parser import parse_order
from ..policy import infer_policy


class MockNormalizerAgent:
    def __init__(self, catalog: list[dict[str, Any]]):
        self.catalog = catalog

    def _candidate_rows(self, parsed: dict[str, Any]) -> list[dict[str, Any]]:
        candidates = [row for row in self.catalog if row["body_part"] == parsed["body_part"]]
        if parsed["laterality"] == "bilateral":
            return []
        if parsed["laterality"] in {"left", "right"}:
            candidates = [row for row in candidates if row["laterality"] in {parsed["laterality"], "none"}]

        filtered: list[dict[str, Any]] = []
        for row in candidates:
            if parsed["portable"] is True and row["portable"] is False:
                continue
            if parsed.get("projection_conflict") and row["body_part"] == "chest":
                continue
            if parsed["projection"] is not None and row["projection"] is not None and parsed["projection"] != row["projection"]:
                continue
            if parsed["subtype"] is not None and row.get("subtype") != parsed["subtype"]:
                continue
            if parsed["view_count"] is not None:
                row_view = row.get("view_count")
                if row["canonical_id"] == "XR_PELVIS_1_2V":
                    if parsed["view_count"] not in {1, 2}:
                        continue
                elif row_view != parsed["view_count"]:
                    continue
            if parsed["view_pattern"] == "one_or_two" and row["canonical_id"] != "XR_PELVIS_1_2V":
                continue
            filtered.append(row)
        return filtered

    def predict(self, *, active_system_prompt: str, order_text: str, runtime_catalog: list[dict[str, Any]]) -> dict[str, Any]:
        del runtime_catalog
        policy = infer_policy(active_system_prompt)
        parsed = parse_order(order_text)
        source_lower = order_text.lower()

        if parsed["modality"] != "XR":
            return {"decision": "needs_review", "canonical_id": None, "reason": "out_of_scope_modality"}
        if parsed["body_part"] is None:
            return {"decision": "needs_review", "canonical_id": None, "reason": "unknown_body_part"}

        candidates = self._candidate_rows(parsed)
        if len(candidates) != 1:
            return {"decision": "needs_review", "canonical_id": None, "reason": "ambiguous_or_no_candidate"}

        row = candidates[0]
        code = row["canonical_id"]
        body_part = row["body_part"]

        if body_part in {"knee", "hand", "ankle"} and policy.strict_extremity:
            if parsed["laterality"] not in {"left", "right"} or parsed["view_count"] != row["view_count"]:
                return {"decision": "needs_review", "canonical_id": None, "reason": "missing_required_extremity_detail"}

        if body_part == "chest" and policy.strict_chest:
            if code == "XR_CHEST_AP_PORTABLE_1V":
                has_core_support = parsed["portable"] is True and parsed["projection"] == "AP" and not parsed.get("projection_conflict")
                if not has_core_support:
                    return {"decision": "needs_review", "canonical_id": None, "reason": "missing_required_chest_detail"}
                if parsed["view_count"] != 1:
                    if not policy.allow_commit_when_unique:
                        return {"decision": "needs_review", "canonical_id": None, "reason": "missing_required_chest_detail"}
                    shorthand_like = any(token in source_lower for token in ["cxr", "rx ", "bedside", "portable cxr"])
                    if policy.chest_shorthand_exception and not shorthand_like:
                        return {"decision": "needs_review", "canonical_id": None, "reason": "missing_required_chest_detail"}
            if code == "XR_CHEST_PA_LAT_2V":
                has_core_support = parsed["projection"] == "PA+LAT" and parsed["portable"] is not True
                if not has_core_support:
                    return {"decision": "needs_review", "canonical_id": None, "reason": "missing_required_chest_detail"}
                if parsed["view_count"] != 2 and not policy.allow_commit_when_unique:
                    return {"decision": "needs_review", "canonical_id": None, "reason": "missing_required_chest_detail"}

        if body_part == "abdomen" and policy.strict_abdomen:
            if not (parsed["subtype"] == "KUB" and parsed["view_count"] == 1 and parsed["portable"] is not True):
                return {"decision": "needs_review", "canonical_id": None, "reason": "missing_required_abdomen_detail"}

        if body_part == "pelvis" and policy.strict_pelvis:
            if not (parsed["view_pattern"] == "one_or_two" and parsed["portable"] is not True):
                return {"decision": "needs_review", "canonical_id": None, "reason": "missing_required_pelvis_detail"}

        return {"decision": "match", "canonical_id": code, "reason": "single_candidate_match"}


class MockFailureAnalyzerAgent:
    def analyze_failure(
        self,
        *,
        current_system_prompt: str,
        order_text: str,
        runtime_catalog: list[dict[str, Any]],
        model_output: dict[str, Any],
        expected_output: dict[str, Any],
        error_class: str,
        expected_attributes: dict[str, Any] | None,
    ) -> dict[str, Any]:
        del current_system_prompt, runtime_catalog, model_output, expected_output

        expected_attributes = expected_attributes or {}
        body_part = expected_attributes.get("body_part")
        source_lower = order_text.lower()

        if error_class == "wrong_match_extremity":
            return {"failure_category": "underspecified_extremity_match", "failure_rationale": "The model matched a side-specific extremity study even though required qualifiers such as laterality or view count were not fully supported.", "policy_gap": "The prompt does not force abstention when extremity qualifiers are missing.", "generalizable_fix_hint": "Require explicit laterality and explicit view count before matching side-specific extremity rows.", "leakage_safe_summary": f"Extremity order was over-matched despite missing qualifiers for {body_part or 'extremity'}.", "severity": "high"}
        if error_class == "wrong_match_chest":
            if "portable" in source_lower and "ap" in source_lower and not any(tok in source_lower for tok in ["1v", "1 view", "one view"]) and not any(tok in source_lower for tok in ["cxr", "rx ", "bedside"]):
                return {"failure_category": "generic_portable_ap_chest_overmatch", "failure_rationale": "The model treated a generic portable AP chest order as if shorthand implied the specific one-view portable chest study.", "policy_gap": "The prompt lacks a rule distinguishing terse shorthand portable chest orders from generic portable AP chest wording when view count is absent.", "generalizable_fix_hint": "Allow missing view count only for terse portable chest shorthand or bedside wording; otherwise require explicit one-view support.", "leakage_safe_summary": "A generic portable AP chest wording was over-matched even though view count was absent.", "severity": "medium"}
            return {"failure_category": "underspecified_chest_match", "failure_rationale": "The model selected a chest study variant without enough support for portability, projection, or view count.", "policy_gap": "The prompt allows chest-specific qualifiers to be inferred too aggressively.", "generalizable_fix_hint": "Require explicit support for portability, chest projection, and view count before matching special chest variants.", "leakage_safe_summary": "Chest order was over-matched despite missing or conflicting technique qualifiers.", "severity": "high"}
        if error_class == "wrong_match_abdomen":
            return {"failure_category": "underspecified_abdomen_match", "failure_rationale": "The model collapsed a generic abdomen order into a more specific abdomen variant without explicit support.", "policy_gap": "The prompt does not clearly forbid subtype inference for abdomen studies.", "generalizable_fix_hint": "Require explicit subtype and explicit view count before matching a specific abdomen variant.", "leakage_safe_summary": "Abdomen order was normalized to an overly specific subtype without direct textual support.", "severity": "high"}
        if error_class == "wrong_match_pelvis":
            return {"failure_category": "underspecified_pelvis_match", "failure_rationale": "The model chose a pelvis variant when the order text did not uniquely support the catalog row qualifiers.", "policy_gap": "The prompt is too permissive for pelvis variants with special view constraints.", "generalizable_fix_hint": "Require explicit support for the pelvis row qualifiers; otherwise abstain.", "leakage_safe_summary": "Pelvis order was over-matched despite incomplete view-pattern support.", "severity": "medium"}
        if error_class == "missed_match":
            return {"failure_category": "over_abstention", "failure_rationale": "The model abstained even though the order provided enough support for a unique match.", "policy_gap": "The prompt may be too cautious or may not explain how to commit when exactly one row remains after filtering.", "generalizable_fix_hint": "Clarify that a match is appropriate when exactly one row remains after applying all explicit constraints.", "leakage_safe_summary": "Model abstained even though the order provided enough evidence for a unique in-scope match.", "severity": "medium"}
        if error_class == "out_of_scope_miss":
            return {"failure_category": "out_of_scope_handling", "failure_rationale": "The model failed to abstain on a non-XR or otherwise out-of-scope order.", "policy_gap": "The prompt does not enforce out-of-scope abstention strongly enough.", "generalizable_fix_hint": "Add an explicit rule to return needs_review for out-of-scope modalities or tasks.", "leakage_safe_summary": "An out-of-scope order should have been rejected earlier.", "severity": "high"}
        return {"failure_category": "other_failure", "failure_rationale": "The model output differs from the expected output because the current prompt leaves an important decision rule underspecified.", "policy_gap": "The prompt needs a clearer tie-breaking or abstention rule.", "generalizable_fix_hint": "Add a more explicit decision rule for ambiguous or underspecified cases.", "leakage_safe_summary": "A general decision rule was missing or too weak.", "severity": "medium"}


class MockPromptOptimizerAgent:
    def propose_prompt(self, *, current_system_prompt: str, optimizer_payload: dict[str, Any]) -> dict[str, Any]:
        lower = current_system_prompt.lower()
        new_prompt = current_system_prompt.strip()
        edits: list[str] = []
        rationale_parts: list[str] = []
        category_counts = optimizer_payload["current_metrics"].get("analyzer_category_counts", {})

        if "return `match` only when the selected runtime catalog row is fully supported by the order text" not in lower:
            new_prompt += ("\nCore rule:\n" "- Return `match` only when the selected runtime catalog row is fully supported by the order text.\n" "- If any required qualifier is missing, conflicting, or shared by multiple plausible rows, return `needs_review`.\n")
            edits.append("Added a global full-support and abstention rule.")
            rationale_parts.append("The current system prompt is too permissive and needs an explicit non-inference anchor.")

        if category_counts.get("underspecified_extremity_match", 0) > 0 and "for extremity studies:" not in lower:
            new_prompt += ("\nFor extremity studies:\n" "- Do not infer laterality.\n" "- Do not infer view count.\n" "- Match a side-specific extremity row only when the order text explicitly supports the side and the required view count.\n" "- Otherwise return `needs_review`.\n")
            edits.append("Added strict extremity qualification rules.")
            rationale_parts.append("Repeated failures show that underspecified extremity orders are being over-matched.")
        elif category_counts.get("underspecified_chest_match", 0) > 0 and "for chest studies:" not in lower:
            new_prompt += ("\nFor chest studies:\n" "- Do not infer portability.\n" "- Do not infer chest projection or view count for special chest variants.\n" "- Match a special chest row only when the order text explicitly supports its portability, projection, and view-count constraints.\n" "- Otherwise return `needs_review`.\n")
            edits.append("Added strict chest qualification rules.")
            rationale_parts.append("Chest failures show over-matching when portability, projection, or view count are incomplete.")
        elif category_counts.get("underspecified_abdomen_match", 0) > 0 and "for specific abdomen variants:" not in lower:
            new_prompt += ("\nFor specific abdomen variants:\n" "- Do not infer subtype.\n" "- Do not infer view count.\n" "- Match a specific abdomen variant only when the order text explicitly supports the subtype and required view count.\n" "- Otherwise return `needs_review`.\n")
            edits.append("Added strict abdomen subtype rules.")
            rationale_parts.append("Abdomen failures show that generic abdomen orders are being collapsed into specific subtypes.")
        elif category_counts.get("underspecified_pelvis_match", 0) > 0 and "for pelvis studies:" not in lower:
            new_prompt += ("\nFor pelvis studies:\n" "- Match a pelvis variant only when the order text explicitly supports the view-pattern constraints required by that row.\n" "- If the pelvis order is generic or conflicts with the row qualifiers, return `needs_review`.\n")
            edits.append("Added pelvis-specific qualifier rules.")
            rationale_parts.append("Pelvis failures show that generic pelvis orders cannot safely map to constrained pelvis variants.")
        elif category_counts.get("over_abstention", 0) > 0 and "when exactly one runtime catalog row remains" not in lower:
            new_prompt += ("\nCommit rule:\n" "- When exactly one runtime catalog row remains after applying all explicit constraints from the order text, return `match`.\n")
            edits.append("Added a commit rule to reduce over-abstention.")
            rationale_parts.append("Some failures show that the prompt may be too cautious even when a unique supported row remains.")
        elif category_counts.get("generic_portable_ap_chest_overmatch", 0) > 0 and "portable ap chest shorthand" not in lower:
            new_prompt += ("\nPortable AP chest shorthand:\n" "- If a portable AP chest order lacks an explicit view count, allow a match only for terse portable AP chest shorthand or bedside language.\n" "- Otherwise return `needs_review`.\n")
            edits.append("Added a shorthand-only exception for portable AP chest orders without explicit view count.")
            rationale_parts.append("One remaining failure shows that generic portable AP chest wording should not be treated the same as terse shorthand or bedside wording.")
        elif category_counts.get("out_of_scope_handling", 0) > 0 and "out-of-scope rule:" not in lower:
            new_prompt += ("\nOut-of-scope rule:\n" "- If the order is not an in-scope radiography order or does not map to the runtime catalog task, return `needs_review`.\n")
            edits.append("Added an explicit out-of-scope rule.")
            rationale_parts.append("Some failures show that non-target modalities or tasks need stronger rejection instructions.")

        if not edits:
            rationale_parts.append("No safe, non-leaking prompt revision was identified from the analyzed failures.")

        return {"new_system_prompt": new_prompt.strip() + "\n", "rationale": " ".join(rationale_parts), "edits_applied": edits}
