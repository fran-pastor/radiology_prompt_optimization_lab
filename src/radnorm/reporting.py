from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_accuracy_evolution(path: Path, trace: list[dict[str, Any]]) -> None:
    labels = [item["label"] for item in trace]
    values = [item["accuracy"] * 100 for item in trace]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.plot(range(len(values)), values, marker="o", linewidth=2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 102)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy evolution across prompt-optimization iterations")
    ax.grid(True, alpha=0.35)
    for idx, value in enumerate(values):
        ax.annotate(f"{value:.1f}%", (idx, value), textcoords="offset points", xytext=(0, 8), ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_executive_summary(trace: list[dict[str, Any]], run_meta: dict[str, Any], benchmark_profile: dict[str, Any]) -> str:
    baseline = trace[0]
    final = trace[-1]
    lines = [
        "# Executive summary",
        "",
        f"- Runtime: **{run_meta['runtime']}**",
        f"- Normalizer model: **{run_meta['normalizer_model']}** (`reasoning_effort={run_meta['normalizer_reasoning_effort']}`)",
        f"- Analyzer model: **{run_meta['analyzer_model']}** (`reasoning_effort={run_meta['analyzer_reasoning_effort']}`)",
        f"- Optimizer model: **{run_meta['optimizer_model']}** (`reasoning_effort={run_meta['optimizer_reasoning_effort']}`)",
        f"- Dataset size: **{benchmark_profile['dataset_size']}**",
        f"- Baseline accuracy: **{baseline['accuracy'] * 100:.1f}%**",
        f"- Final accuracy: **{final['accuracy'] * 100:.1f}%**",
        f"- Absolute gain: **{(final['accuracy'] - baseline['accuracy']) * 100:.1f} points**",
        f"- Max iterations configured: **{run_meta['max_iterations']}**",
        "",
        "## Architecture",
        "",
        "1. `NormalizerAgent` consumes the full current system prompt and only the case data in the user input.",
        "2. `FailureAnalyzerAgent` inspects each failed case and produces a short generalizable diagnosis.",
        "3. `PromptOptimizerAgent` receives the current prompt plus analyzed failures and rewrites the full system prompt.",
        "4. Deterministic evaluation decides whether the candidate prompt is accepted.",
    ]
    return "\n".join(lines)

def update_live_accuracy_plot(
    path: Path,
    accepted_points: list[tuple[int, float]],
    discarded_points: list[tuple[int, float]],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.6))

    if accepted_points:
        xs, ys = zip(*accepted_points)
        ys_pct = [y * 100 for y in ys]
        ax.plot(xs, ys_pct, marker="o", linewidth=2, label="Accepted prompt")

        for x, y in zip(xs, ys_pct):
            ax.annotate(
                f"{y:.1f}%",
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
            )

    if discarded_points:
        dx, dy = zip(*discarded_points)
        dy_pct = [y * 100 for y in dy]
        ax.scatter(dx, dy_pct, marker="x", s=90, label="Discarded candidate")

        for x, y in zip(dx, dy_pct):
            ax.annotate(
                f"{y:.1f}%",
                (x, y),
                textcoords="offset points",
                xytext=(0, -14),
                ha="center",
            )

    max_x = 0
    if accepted_points:
        max_x = max(max_x, max(x for x, _ in accepted_points))
    if discarded_points:
        max_x = max(max_x, max(x for x, _ in discarded_points))

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Prompt optimization progress")
    ax.set_xlim(-0.25, max_x + 0.5)
    ax.set_ylim(0, 102)
    ax.grid(True, alpha=0.35)
    ax.legend()

    fig.tight_layout()

    tmp_path = path.parent / f"{path.name}.tmp.png"
    fig.savefig(tmp_path, dpi=180)
    plt.close(fig)
    os.replace(tmp_path, path)
