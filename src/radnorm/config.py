from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    runtime: str
    openai_api_key: str | None
    normalizer_model: str
    normalizer_reasoning_effort: str
    analyzer_model: str
    analyzer_reasoning_effort: str
    optimizer_model: str
    optimizer_reasoning_effort: str
    max_iterations: int
    target_accuracy: float
    patience: int
    debug: bool
    dataset_path: str
    catalog_path: str
    max_failures_for_optimizer: int


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_settings(project_root: Path) -> Settings:
    load_dotenv(project_root / ".env", override=False)
    return Settings(
        runtime=os.getenv("RADNORM_RUNTIME", "mock").strip().lower(),
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        normalizer_model=os.getenv("RADNORM_NORMALIZER_MODEL", "gpt-5-mini"),
        normalizer_reasoning_effort=os.getenv("RADNORM_NORMALIZER_REASONING_EFFORT", "low"),
        analyzer_model=os.getenv("RADNORM_ANALYZER_MODEL", "gpt-5-mini"),
        analyzer_reasoning_effort=os.getenv("RADNORM_ANALYZER_REASONING_EFFORT", "low"),
        optimizer_model=os.getenv("RADNORM_OPTIMIZER_MODEL", "gpt-5.1"),
        optimizer_reasoning_effort=os.getenv("RADNORM_OPTIMIZER_REASONING_EFFORT", "high"),
        max_iterations=int(os.getenv("RADNORM_MAX_ITERATIONS", "100")),
        target_accuracy=float(os.getenv("RADNORM_TARGET_ACCURACY", "1.0")),
        patience=int(os.getenv("RADNORM_PATIENCE", "8")),
        debug=_to_bool(os.getenv("RADNORM_DEBUG"), False),
        dataset_path=os.getenv("RADNORM_DATASET_PATH", "benchmark/dev.jsonl"),
        catalog_path=os.getenv("RADNORM_CATALOG_PATH", "benchmark/catalog.json"),
        max_failures_for_optimizer=int(os.getenv("RADNORM_MAX_FAILURES_FOR_OPTIMIZER", "25")),
    )
