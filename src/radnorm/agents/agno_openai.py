from __future__ import annotations

import json
import os
import time
from typing import Any

from ..config import Settings
from ..schemas import FailureAnalysis, NormalizationResult, PromptProposal


def _clip(text: str, size: int = 140) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= size else text[: size - 3] + "..."


class AgnoOpenAINormalizerAgent:
    def __init__(self, settings: Settings, logger: Any | None = None):
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for agno_openai runtime.")
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        from agno.agent import Agent
        from agno.models.openai import OpenAIResponses

        self.debug = settings.debug
        self.logger = logger
        self.agent = Agent(
            model=OpenAIResponses(id=settings.normalizer_model, reasoning_effort=settings.normalizer_reasoning_effort),
            system_message="placeholder",
            output_schema=NormalizationResult,
            structured_outputs=True,
            markdown=False,
            debug_mode=self.debug,
        )

    def predict(self, *, active_system_prompt: str, order_text: str, runtime_catalog: list[dict[str, Any]]) -> dict[str, Any]:
        user_message = (
            "<ORDER_TEXT>\n"
            f"{order_text}\n"
            "</ORDER_TEXT>\n\n"
            "<RUNTIME_CATALOG_JSON>\n"
            f"{json.dumps(runtime_catalog, ensure_ascii=False)}\n"
            "</RUNTIME_CATALOG_JSON>"
        )
        if self.debug and self.logger:
            self.logger.debug(
                "[normalizer] dispatch | order_chars=%d | prompt_chars=%d | order_preview=%s",
                len(order_text),
                len(active_system_prompt),
                _clip(order_text),
            )
        started = time.perf_counter()
        self.agent.system_message = active_system_prompt
        result = self.agent.run(user_message, debug_mode=self.debug)
        elapsed = time.perf_counter() - started
        content = result.content
        if hasattr(content, "model_dump"):
            content = content.model_dump()
        if self.debug and self.logger:
            self.logger.debug(
                "[normalizer] done | elapsed=%.2fs | decision=%s | canonical_id=%s",
                elapsed,
                content.get("decision") if isinstance(content, dict) else None,
                content.get("canonical_id") if isinstance(content, dict) else None,
            )
        if isinstance(content, dict):
            return content
        raise RuntimeError("Unexpected Agno normalizer output format.")


class AgnoOpenAIFailureAnalyzerAgent:
    def __init__(self, settings: Settings, system_message: str, logger: Any | None = None):
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for agno_openai runtime.")
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        from agno.agent import Agent
        from agno.models.openai import OpenAIResponses

        self.debug = settings.debug
        self.logger = logger
        self.agent = Agent(
            model=OpenAIResponses(id=settings.analyzer_model, reasoning_effort=settings.analyzer_reasoning_effort),
            system_message=system_message,
            output_schema=FailureAnalysis,
            structured_outputs=True,
            markdown=False,
            debug_mode=self.debug,
        )

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
        user_message = (
            "<CURRENT_SYSTEM_PROMPT>\n"
            f"{current_system_prompt.strip()}\n"
            "</CURRENT_SYSTEM_PROMPT>\n\n"
            "<NORMALIZER_USER_INPUT>\n"
            f"{order_text}\n"
            "</NORMALIZER_USER_INPUT>\n\n"
            "<RUNTIME_CATALOG_JSON>\n"
            f"{json.dumps(runtime_catalog, ensure_ascii=False)}\n"
            "</RUNTIME_CATALOG_JSON>\n\n"
            "<NORMALIZER_OUTPUT_JSON>\n"
            f"{json.dumps(model_output, ensure_ascii=False)}\n"
            "</NORMALIZER_OUTPUT_JSON>\n\n"
            "<EXPECTED_OUTPUT_JSON>\n"
            f"{json.dumps(expected_output, ensure_ascii=False)}\n"
            "</EXPECTED_OUTPUT_JSON>\n\n"
            "<EVALUATION_CONTEXT_JSON>\n"
            f"{json.dumps({'error_class': error_class, 'expected_attributes': expected_attributes or {}}, ensure_ascii=False)}\n"
            "</EVALUATION_CONTEXT_JSON>"
        )
        if self.debug and self.logger:
            self.logger.debug("[analyzer] dispatch | error_class=%s | order_preview=%s", error_class, _clip(order_text))
        started = time.perf_counter()
        result = self.agent.run(user_message, debug_mode=self.debug)
        elapsed = time.perf_counter() - started
        content = result.content
        if hasattr(content, "model_dump"):
            content = content.model_dump()
        if self.debug and self.logger:
            self.logger.debug(
                "[analyzer] done | elapsed=%.2fs | category=%s",
                elapsed,
                content.get("failure_category") if isinstance(content, dict) else None,
            )
        if isinstance(content, dict):
            return content
        raise RuntimeError("Unexpected Agno analyzer output format.")


class AgnoOpenAIPromptOptimizerAgent:
    def __init__(self, settings: Settings, system_message: str, logger: Any | None = None):
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for agno_openai runtime.")
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        from agno.agent import Agent
        from agno.models.openai import OpenAIResponses

        self.debug = settings.debug
        self.logger = logger
        self.agent = Agent(
            model=OpenAIResponses(id=settings.optimizer_model, reasoning_effort=settings.optimizer_reasoning_effort),
            system_message=system_message,
            output_schema=PromptProposal,
            structured_outputs=True,
            markdown=False,
            debug_mode=self.debug,
        )

    def propose_prompt(self, *, current_system_prompt: str, optimizer_payload: dict[str, Any]) -> dict[str, Any]:
        user_message = (
            "<CURRENT_SYSTEM_PROMPT>\n"
            f"{current_system_prompt.strip()}\n"
            "</CURRENT_SYSTEM_PROMPT>\n\n"
            "<ANALYZED_FAILURES_JSON>\n"
            f"{json.dumps(optimizer_payload['analyzed_failures'], ensure_ascii=False)}\n"
            "</ANALYZED_FAILURES_JSON>"
        )
        if self.debug and self.logger:
            failure_counts = optimizer_payload.get("current_metrics", {}).get("failure_counts", {})
            self.logger.debug(
                "[optimizer] dispatch | prompt_chars=%d | failure_types=%d | top_failure_keys=%s",
                len(current_system_prompt),
                len(failure_counts),
                list(failure_counts.keys())[:5],
            )
        started = time.perf_counter()
        result = self.agent.run(user_message, debug_mode=self.debug)
        elapsed = time.perf_counter() - started
        content = result.content
        if hasattr(content, "model_dump"):
            content = content.model_dump()
        if self.debug and self.logger:
            self.logger.debug(
                "[optimizer] done | elapsed=%.2fs | edits=%d",
                elapsed,
                len(content.get("edits_applied", [])) if isinstance(content, dict) else 0,
            )
        if isinstance(content, dict):
            return content
        raise RuntimeError("Unexpected Agno optimizer output format.")
