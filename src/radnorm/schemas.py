from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class NormalizationResult(BaseModel):
    decision: Literal["match", "needs_review"]
    canonical_id: str | None = Field(default=None)
    reason: str

    @model_validator(mode="after")
    def validate_consistency(self) -> "NormalizationResult":
        if self.decision == "match" and not self.canonical_id:
            raise ValueError("canonical_id must be non-null when decision is 'match'.")
        if self.decision == "needs_review" and self.canonical_id is not None:
            raise ValueError("canonical_id must be null when decision is 'needs_review'.")
        return self


class FailureAnalysis(BaseModel):
    failure_category: str
    failure_rationale: str
    policy_gap: str
    generalizable_fix_hint: str
    leakage_safe_summary: str
    severity: Literal["low", "medium", "high"] = "medium"


class PromptProposal(BaseModel):
    new_system_prompt: str
    rationale: str
    edits_applied: list[str] = Field(default_factory=list)
