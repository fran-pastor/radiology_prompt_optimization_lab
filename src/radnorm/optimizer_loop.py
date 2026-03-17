from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .benchmark_profile import build_optimizer_payload
from .evaluation import evaluate_predictions
from .leakage_guard import find_leaks
from .reporting import update_live_accuracy_plot


@dataclass(frozen=True)
class LoopConfig:
    max_iterations: int
    target_accuracy: float
    patience: int
    debug: bool
    max_failures_for_optimizer: int


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / 4))


def _top_counts(counts: dict[str, int], limit: int = 5) -> list[tuple[str, int]]:
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]


def _predict_dataset(
    *,
    agent: Any,
    dataset: list[dict[str, Any]],
    active_system_prompt: str,
    runtime_catalog: list[dict[str, Any]],
    logger: Any,
    debug: bool,
    iteration: int,
    phase: str,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    total = len(dataset)
    logger.info(f"[{iteration:03d}] {phase} started | rows={total}")
    process_started = time.perf_counter()

    for idx, row in enumerate(dataset, start=1):
        row_id = row.get("record_id") or row.get("case_id") or f"row_{idx}"
        if debug:
            logger.debug(f"[{iteration:03d}] {phase} row_start | {idx}/{total} | id={row_id}")

        row_started = time.perf_counter()
        prediction = agent.predict(active_system_prompt=active_system_prompt, order_text=row["source_text"], runtime_catalog=runtime_catalog)
        row_elapsed = time.perf_counter() - row_started
        predictions.append(prediction)

        elapsed_total = time.perf_counter() - process_started
        avg = elapsed_total / idx
        remaining = max(total - idx, 0)
        eta = remaining * avg

        if debug or idx == total or idx % 10 == 0:
            logger.info(
                f"[{iteration:03d}] {phase} progress | {idx}/{total} | id={row_id} | row_elapsed={row_elapsed:.2f}s | avg={avg:.2f}s | eta={eta:.1f}s | decision={prediction.get('decision')} | canonical_id={prediction.get('canonical_id')}"
            )

    logger.info(f"[{iteration:03d}] {phase} completed | rows={total}")
    return predictions


def _analyze_failures(
    *,
    analyzer_agent: Any,
    evaluation: dict[str, Any],
    current_system_prompt: str,
    runtime_catalog: list[dict[str, Any]],
    logger: Any,
    debug: bool,
    iteration: int,
) -> list[dict[str, Any]]:
    analyses: list[dict[str, Any]] = []
    failures = evaluation["failures"]
    logger.info(f"[{iteration:03d}] failure_analysis started | failures={len(failures)}")
    started = time.perf_counter()

    for idx, failure in enumerate(failures, start=1):
        record_id = failure["record_id"]
        if debug:
            logger.debug(f"[{iteration:03d}] failure_analysis item_start | {idx}/{len(failures)} | id={record_id} | error_class={failure['error_class']}")
        item_started = time.perf_counter()
        analysis = analyzer_agent.analyze_failure(
            current_system_prompt=current_system_prompt,
            order_text=failure["source_text"],
            runtime_catalog=runtime_catalog,
            model_output=failure["predicted_output"],
            expected_output=failure["expected_output"],
            error_class=failure["error_class"],
            expected_attributes=failure.get("expected_attributes"),
        )
        analyses.append(analysis)
        item_elapsed = time.perf_counter() - item_started
        logger.info(
            f"[{iteration:03d}] failure_analysis progress | {idx}/{len(failures)} | id={record_id} | elapsed={item_elapsed:.2f}s | category={analysis.get('failure_category')}"
        )

    logger.info(f"[{iteration:03d}] failure_analysis completed | failures={len(failures)} | elapsed={time.perf_counter() - started:.2f}s")
    return analyses


def run_optimization_loop(
    *,
    dataset: list[dict[str, Any]],
    runtime_catalog: list[dict[str, Any]],
    benchmark_profile: dict[str, Any],
    normalizer_agent: Any,
    analyzer_agent: Any,
    optimizer_agent: Any,
    starting_system_prompt: str,
    forbidden_literals: set[str],
    loop_config: LoopConfig,
    logger: Any,
    plot_path
) -> dict[str, Any]:
    current_system_prompt = starting_system_prompt
    current_predictions: list[dict[str, Any]] | None = None
    current_evaluation: dict[str, Any] | None = None
    current_analyses: list[dict[str, Any]] | None = None

    no_improvement_rounds = 0
    trace: list[dict[str, Any]] = []
    stage_payloads: list[dict[str, Any]] = []

    logger.info("Starting optimization loop")
    logger.info(f"Dataset size={len(dataset)} | target_accuracy={loop_config.target_accuracy:.3f} | max_iterations={loop_config.max_iterations}")

    accepted_points: list[tuple[int, float]] = []
    discarded_points: list[tuple[int, float]] = []

    for iteration in range(0, loop_config.max_iterations + 1):
        started = time.perf_counter()

        if current_predictions is None or current_evaluation is None:
            current_predictions = _predict_dataset(
                agent=normalizer_agent,
                dataset=dataset,
                active_system_prompt=current_system_prompt,
                runtime_catalog=runtime_catalog,
                logger=logger,
                debug=loop_config.debug,
                iteration=iteration,
                phase="current_prompt_eval",
            )
            current_evaluation = evaluate_predictions(current_predictions, dataset)
            current_analyses = None
        else:
            logger.info(f"[{iteration:03d}] current_prompt_eval skipped | using cached accepted evaluation")

        if current_analyses is None:
            current_analyses = _analyze_failures(
                analyzer_agent=analyzer_agent,
                evaluation=current_evaluation,
                current_system_prompt=current_system_prompt,
                runtime_catalog=runtime_catalog,
                logger=logger,
                debug=loop_config.debug,
                iteration=iteration,
            )
        else:
            logger.info(f"[{iteration:03d}] failure_analysis skipped | using cached accepted analyses")

        elapsed = time.perf_counter() - started
        accuracy = current_evaluation["accuracy"]
        if iteration == 0 and not accepted_points:
            accepted_points.append((0, accuracy))
            update_live_accuracy_plot(plot_path, accepted_points, discarded_points)

        trace_item = {
            "iteration": iteration,
            "label": "baseline" if iteration == 0 else f"iteration_{iteration}",
            "accuracy": accuracy,
            "correct": current_evaluation["correct"],
            "total": current_evaluation["total"],
            "failure_counts": current_evaluation["failure_counts"],
            "prompt_characters": len(current_system_prompt),
            "prompt_token_estimate": estimate_tokens(current_system_prompt),
            "elapsed_seconds": round(elapsed, 3),
            "analysis_count": len(current_analyses),
        }
        trace.append(trace_item)
        stage_payloads.append({
            "iteration": iteration,
            "system_prompt": current_system_prompt,
            "evaluation": current_evaluation,
            "predictions": current_predictions,
            "analyses": current_analyses,
        })

        top_errors = _top_counts(current_evaluation["failure_counts"])
        logger.info(f"[{iteration:03d}] summary | accuracy={accuracy:.4f} ({current_evaluation['correct']}/{current_evaluation['total']}) | prompt_tokens≈{estimate_tokens(current_system_prompt)} | elapsed={elapsed:.2f}s")
        if top_errors:
            logger.info(f"[{iteration:03d}] summary | top_errors=" + ", ".join(f"{name}:{count}" for name, count in top_errors))
        else:
            logger.info(f"[{iteration:03d}] summary | top_errors=none")

        if accuracy >= loop_config.target_accuracy:
            logger.info("Target accuracy reached. Stopping loop.")
            break

        optimizer_payload = build_optimizer_payload(
            benchmark_profile,
            current_evaluation,
            current_analyses,
            loop_config.max_failures_for_optimizer,
        )
        optimizer_payload["loop_context"] = {
            "iteration": iteration + 1,
            "max_iterations": loop_config.max_iterations,
            "current_accuracy": accuracy,
            "current_prompt_token_estimate": estimate_tokens(current_system_prompt),
        }

        logger.info(f"[{iteration:03d}] optimizer | building proposal")
        proposal = optimizer_agent.propose_prompt(current_system_prompt=current_system_prompt, optimizer_payload=optimizer_payload)
        proposed_prompt = proposal.get("new_system_prompt", "").strip()
        trace_item["optimizer_rationale"] = proposal.get("rationale", "")
        trace_item["edits_applied"] = proposal.get("edits_applied", [])
        trace_item["proposed_system_prompt"] = proposed_prompt
        logger.info(f"[{iteration:03d}] optimizer | proposal_received | edits={len(trace_item['edits_applied'])} | new_prompt_chars={len(proposed_prompt)}")

        leaks = find_leaks(proposed_prompt, forbidden_literals)
        if leaks:
            logger.info(f"[{iteration:03d}] optimizer | proposal_rejected=leakage_guard | hits={len(leaks)}")
            trace_item["accepted_next_prompt"] = False
            trace_item["rejection_reason"] = "leakage_guard"
            trace_item["leak_hits"] = leaks
            no_improvement_rounds += 1
            if no_improvement_rounds >= loop_config.patience:
                logger.info("Patience exhausted after repeated non-improving rounds.")
                break
            continue

        candidate_predictions = _predict_dataset(
            agent=normalizer_agent,
            dataset=dataset,
            active_system_prompt=proposed_prompt,
            runtime_catalog=runtime_catalog,
            logger=logger,
            debug=loop_config.debug,
            iteration=iteration,
            phase="candidate_prompt_eval",
        )
        candidate_evaluation = evaluate_predictions(candidate_predictions, dataset)
        candidate_accuracy = candidate_evaluation["accuracy"]
        candidate_iteration = iteration + 1
        trace_item["candidate_accuracy"] = candidate_accuracy
        logger.info(f"[{iteration:03d}] optimizer | candidate_accuracy={candidate_accuracy:.4f} | current_accuracy={accuracy:.4f}")

        if candidate_accuracy > accuracy:
            accepted_points.append((candidate_iteration, candidate_accuracy))
            update_live_accuracy_plot(plot_path, accepted_points, discarded_points)
            current_system_prompt = proposed_prompt
            current_predictions = candidate_predictions
            current_evaluation = candidate_evaluation
            current_analyses = None
            trace_item["accepted_next_prompt"] = True
            no_improvement_rounds = 0
            logger.info(f"[{iteration:03d}] optimizer | proposal_accepted | gain={candidate_accuracy - accuracy:.4f}")
            if loop_config.debug:
                logger.debug(f"[{iteration:03d}] optimizer_rationale={proposal.get('rationale', '')}")
        else:
            discarded_points.append((candidate_iteration, candidate_accuracy))
            update_live_accuracy_plot(plot_path, accepted_points, discarded_points)
            trace_item["accepted_next_prompt"] = False
            trace_item["rejection_reason"] = "no_improvement"
            no_improvement_rounds += 1
            logger.info(f"[{iteration:03d}] optimizer | proposal_rejected=no_improvement")
            if loop_config.debug:
                logger.debug(f"[{iteration:03d}] optimizer_rationale={proposal.get('rationale', '')}")
            if no_improvement_rounds >= loop_config.patience:
                logger.info("Patience exhausted after repeated non-improving rounds.")
                break

    return {
        "trace": trace,
        "stage_payloads": stage_payloads,
        "final_system_prompt": current_system_prompt,
        "accepted_plot_points": accepted_points,
        "discarded_plot_points": discarded_points,
    }
