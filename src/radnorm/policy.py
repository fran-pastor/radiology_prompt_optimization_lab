from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptPolicy:
    require_full_support: bool = False
    strict_extremity: bool = False
    strict_chest: bool = False
    strict_abdomen: bool = False
    strict_pelvis: bool = False
    allow_commit_when_unique: bool = False
    chest_shorthand_exception: bool = False


def infer_policy(prompt_text: str) -> PromptPolicy:
    lower = prompt_text.lower()
    return PromptPolicy(
        require_full_support=("fully supported by the order text" in lower),
        strict_extremity=("for extremity studies:" in lower and "do not infer laterality" in lower and "do not infer view count" in lower),
        strict_chest=("for chest studies:" in lower and "do not infer portability" in lower and "do not infer chest projection or view count" in lower),
        strict_abdomen=("for specific abdomen variants:" in lower and "do not infer subtype" in lower and "do not infer view count" in lower),
        strict_pelvis=("for pelvis studies:" in lower and "view-pattern constraints" in lower),
        allow_commit_when_unique=("when exactly one runtime catalog row remains" in lower),
        chest_shorthand_exception=("portable ap chest shorthand" in lower and "bedside language" in lower),
    )
