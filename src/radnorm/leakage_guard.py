from __future__ import annotations


def build_forbidden_literals(dataset: list[dict], catalog: list[dict]) -> set[str]:
    literals: set[str] = set()
    for row in dataset:
        text = row["source_text"].strip().lower()
        if len(text) >= 18 and len(text.split()) >= 3:
            literals.add(text)
    for row in catalog:
        code = row.get("canonical_id")
        name = row.get("canonical_name")
        if isinstance(code, str) and len(code.strip()) >= 6:
            literals.add(code.strip().lower())
        if isinstance(name, str) and len(name.strip()) >= 18 and len(name.split()) >= 3:
            literals.add(name.strip().lower())
    return literals


def find_leaks(prompt_text: str, forbidden_literals: set[str]) -> list[str]:
    lower = prompt_text.lower()
    hits = [literal for literal in forbidden_literals if literal in lower]
    return sorted(hits)[:20]
