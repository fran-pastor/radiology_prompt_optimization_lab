from __future__ import annotations

from typing import Any, Protocol


class SupportsNormalize(Protocol):
    def predict(
        self,
        *,
        active_system_prompt: str,
        order_text: str,
        runtime_catalog: list[dict[str, Any]],
    ) -> dict[str, Any]: ...


class SupportsAnalyzeFailure(Protocol):
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
    ) -> dict[str, Any]: ...


class SupportsOptimizePrompt(Protocol):
    def propose_prompt(
        self,
        *,
        current_system_prompt: str,
        optimizer_payload: dict[str, Any],
    ) -> dict[str, Any]: ...