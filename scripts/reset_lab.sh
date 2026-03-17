#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[reset] Restoring baseline system prompt"
cp "$ROOT/prompts/normalizer/baseline_system_prompt.txt" "$ROOT/prompts/normalizer/current_system_prompt.txt"

echo "[reset] Cleaning prompt history"
rm -rf "$ROOT/prompts/normalizer/accepted" "$ROOT/prompts/normalizer/candidates"
mkdir -p "$ROOT/prompts/normalizer/accepted" "$ROOT/prompts/normalizer/candidates"

echo "[reset] Cleaning results"
rm -rf "$ROOT/results"
mkdir -p "$ROOT/results/generated_prompts"

echo "[reset] Removing derived optimized prompt"
rm -f "$ROOT/prompts/optimized_prompt_final.txt"

echo "[reset] Done"
