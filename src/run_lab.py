from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from radnorm.agents import build_agents
from radnorm.benchmark import load_catalog, load_jsonl, load_text, project_root, write_text
from radnorm.benchmark_profile import build_benchmark_profile
from radnorm.config import load_settings
from radnorm.leakage_guard import build_forbidden_literals
from radnorm.logging_utils import build_logger
from radnorm.optimizer_loop import LoopConfig, run_optimization_loop
from radnorm.reporting import build_executive_summary, plot_accuracy_evolution, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the three-agent radiology prompt-optimization lab.")
    parser.add_argument("--runtime", choices=["mock", "agno_openai"], default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=None)
    args = parser.parse_args()

    root = project_root()
    settings = load_settings(root)
    if args.runtime is not None:
        settings = replace(settings, runtime=args.runtime)
    if args.debug:
        settings = replace(settings, debug=True)
    if args.max_iterations is not None:
        settings = replace(settings, max_iterations=args.max_iterations)

    results_dir = root / "results"
    generated_dir = results_dir / "generated_prompts"
    accepted_dir = root / "prompts" / "normalizer" / "accepted"
    candidates_dir = root / "prompts" / "normalizer" / "candidates"
    for path in [results_dir, generated_dir, accepted_dir, candidates_dir]:
        path.mkdir(parents=True, exist_ok=True)

    for stale in results_dir.glob("predictions_iteration_*.json"):
        stale.unlink()
    for stale in generated_dir.glob("prompt_iteration_*.txt"):
        stale.unlink()
    for stale in accepted_dir.glob("*.txt"):
        stale.unlink()
    for stale in candidates_dir.glob("*.txt"):
        stale.unlink()

    logger = build_logger(settings.debug, results_dir / "debug_log.txt")

    dataset = load_jsonl(root / settings.dataset_path)
    catalog = load_catalog(root / settings.catalog_path)
    starting_system_prompt = load_text(root / "prompts" / "normalizer" / "current_system_prompt.txt")
    baseline_system_prompt = load_text(root / "prompts" / "normalizer" / "baseline_system_prompt.txt")
    analyzer_system_message = load_text(root / "prompts" / "analyzer" / "system_message.txt")
    optimizer_system_message = load_text(root / "prompts" / "optimizer" / "system_message.txt")

    benchmark_profile = build_benchmark_profile(dataset)
    write_json(results_dir / "benchmark_profile.json", benchmark_profile)
    forbidden_literals = build_forbidden_literals(dataset, catalog)

    logger.info(f"Runtime={settings.runtime}")
    logger.info(
        f"Normalizer model={settings.normalizer_model} (reasoning_effort={settings.normalizer_reasoning_effort}) | "
        f"Analyzer model={settings.analyzer_model} (reasoning_effort={settings.analyzer_reasoning_effort}) | "
        f"Optimizer model={settings.optimizer_model} (reasoning_effort={settings.optimizer_reasoning_effort})"
    )
    logger.info(f"Debug mode={'on' if settings.debug else 'off'}")
    logger.info(f"Starting prompt source={'current_system_prompt.txt' if starting_system_prompt != baseline_system_prompt else 'baseline_system_prompt.txt'}")

    normalizer_agent, analyzer_agent, optimizer_agent = build_agents(
        settings,
        catalog=catalog,
        analyzer_system_message=analyzer_system_message,
        optimizer_system_message=optimizer_system_message,
        logger=logger,
    )

    outcome = run_optimization_loop(
        dataset=dataset,
        runtime_catalog=catalog,
        benchmark_profile=benchmark_profile,
        normalizer_agent=normalizer_agent,
        analyzer_agent=analyzer_agent,
        optimizer_agent=optimizer_agent,
        starting_system_prompt=starting_system_prompt,
        forbidden_literals=forbidden_literals,
        loop_config=LoopConfig(
            max_iterations=settings.max_iterations,
            target_accuracy=settings.target_accuracy,
            patience=settings.patience,
            debug=settings.debug,
            max_failures_for_optimizer=settings.max_failures_for_optimizer,
        ),
        logger=logger,
        plot_path=results_dir / "accuracy_evolution.png",
    )

    trace = outcome["trace"]
    stage_payloads = outcome["stage_payloads"]
    final_system_prompt = outcome["final_system_prompt"]

    for stage in stage_payloads:
        idx = stage["iteration"]
        write_json(results_dir / f"predictions_iteration_{idx:03d}.json", stage)
        write_text(generated_dir / f"prompt_iteration_{idx:03d}.txt", stage["system_prompt"])
        write_text(accepted_dir / f"iter_{idx:03d}.txt", stage["system_prompt"])

    for item in trace:
        if item.get("proposed_system_prompt"):
            write_text(candidates_dir / f"iter_{item['iteration']:03d}_candidate.txt", item["proposed_system_prompt"])

    write_text(root / "prompts" / "normalizer" / "current_system_prompt.txt", final_system_prompt)
    write_text(root / "prompts" / "optimized_prompt_final.txt", final_system_prompt)

    write_json(
        results_dir / "optimization_trace.json",
        {
            "settings": {
                "runtime": settings.runtime,
                "normalizer_model": settings.normalizer_model,
                "normalizer_reasoning_effort": settings.normalizer_reasoning_effort,
                "analyzer_model": settings.analyzer_model,
                "analyzer_reasoning_effort": settings.analyzer_reasoning_effort,
                "optimizer_model": settings.optimizer_model,
                "optimizer_reasoning_effort": settings.optimizer_reasoning_effort,
                "max_iterations": settings.max_iterations,
                "target_accuracy": settings.target_accuracy,
                "patience": settings.patience,
                "debug": settings.debug,
            },
            "trace": trace,
        },
    )
    #plot_accuracy_evolution(results_dir / "accuracy_evolution.png", trace)
    summary = build_executive_summary(
        trace,
        {
            "runtime": settings.runtime,
            "normalizer_model": settings.normalizer_model,
            "normalizer_reasoning_effort": settings.normalizer_reasoning_effort,
            "analyzer_model": settings.analyzer_model,
            "analyzer_reasoning_effort": settings.analyzer_reasoning_effort,
            "optimizer_model": settings.optimizer_model,
            "optimizer_reasoning_effort": settings.optimizer_reasoning_effort,
            "max_iterations": settings.max_iterations,
        },
        benchmark_profile,
    )
    write_text(results_dir / "executive_summary.md", summary)

    logger.info("Run complete")
    logger.info(f"Final accuracy={trace[-1]['accuracy']:.4f}")
    logger.info(f"Artifacts written to {results_dir}")


if __name__ == "__main__":
    main()
