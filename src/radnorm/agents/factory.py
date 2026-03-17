from __future__ import annotations

from ..config import Settings
from .agno_openai import AgnoOpenAINormalizerAgent, AgnoOpenAIFailureAnalyzerAgent, AgnoOpenAIPromptOptimizerAgent
from .mock import MockFailureAnalyzerAgent, MockNormalizerAgent, MockPromptOptimizerAgent


def build_agents(
    settings: Settings,
    *,
    catalog: list[dict],
    analyzer_system_message: str,
    optimizer_system_message: str,
    logger=None,
):
    if settings.runtime == "mock":
        return (
            MockNormalizerAgent(catalog),
            MockFailureAnalyzerAgent(),
            MockPromptOptimizerAgent(),
        )
    if settings.runtime == "agno_openai":
        return (
            AgnoOpenAINormalizerAgent(settings, logger=logger),
            AgnoOpenAIFailureAnalyzerAgent(settings, analyzer_system_message, logger=logger),
            AgnoOpenAIPromptOptimizerAgent(settings, optimizer_system_message, logger=logger),
        )
    raise ValueError(f"Unsupported runtime: {settings.runtime}")
