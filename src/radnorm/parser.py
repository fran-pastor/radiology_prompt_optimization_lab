from __future__ import annotations

import re
from typing import Any


NON_XR_MODALITIES = {
    "CT": [r"\bct\b"],
    "MRI": [r"\bmri\b"],
    "US": [r"\bus\b", r"\bultrasound\b"],
    "FLUORO": [r"\bfluoro\b", r"\bfluoroscopy\b"],
}

BODY_PART_PATTERNS = {
    "chest": [r"\bchest\b", r"\bcxr\b", r"\btorax\b", r"\btórax\b", r"thorax"],
    "knee": [r"\bknee\b", r"\brodilla\b"],
    "hand": [r"\bhand\b", r"\bmano\b"],
    "ankle": [r"\bankle\b", r"\btobillo\b"],
    "abdomen": [r"\babdomen\b", r"\babdominal\b", r"\bkub\b", r"acute\s+series", r"upper\s+gi"],
    "pelvis": [r"\bpelvis\b", r"\bpelvic\b"],
}

XR_PATTERNS = [r"\bxr\b", r"\bxray\b", r"x-ray", r"\brx\b", r"radiograph", r"\bradiographs\b", r"\bcxr\b", r"\bfilm\b"]
LEFT_PATTERNS = [r"\bleft\b", r"\blt\b", r"(?:^|\s)l(?=\s)", r"izquierda", r"izquierdo"]
RIGHT_PATTERNS = [r"\bright\b", r"\brt\b", r"(?:^|\s)r(?=\s)", r"derecha", r"derecho"]


def _contains_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _detect_view_count(text: str, body_part: str | None) -> tuple[int | None, str | None]:
    if re.search(r"\b1\s*[-/]\s*2\s*views?\b|\bone\s+or\s+two\s+views?\b|\bone/two[- ]view\b|\b1\s*o\s*2\s*vistas?\b", text):
        return None, "one_or_two"
    patterns = [
        (1, [r"\b1\s*v\b", r"\b1\s*views?\b", r"\bone\s+views?\b", r"\bsingle[- ]views?\b", r"\b1\s*vistas?\b", r"\b1\s*vista\b", r"\bone-view\b"]),
        (2, [r"\b2\s*v\b", r"\b2\s*views?\b", r"\b2[- ]view\b", r"\btwo\s+views?\b", r"\b2\s*vistas?\b", r"\b2\s*proyecciones\b", r"\btwo-view\b"]),
        (3, [r"\b3\s*v\b", r"\b3\s*views?\b", r"\b3[- ]view\b", r"\bthree\s+views?\b", r"\bthree-view\b", r"\b3\s*vistas?\b", r"\b3\s*proyecciones\b"]),
    ]
    for count, pats in patterns:
        if _contains_any(text, pats):
            return count, None
    if body_part == "hand" and _contains_any(text, [r"\bap\b"]) and _contains_any(text, [r"\boblique\b", r"\bobl\b"]) and _contains_any(text, [r"lateral"]):
        return 3, None
    if body_part == "ankle" and _contains_any(text, [r"\bap\b"]) and _contains_any(text, [r"mortise"]) and _contains_any(text, [r"lateral"]):
        return 3, None
    return None, None


def parse_order(text: str) -> dict[str, Any]:
    original = text
    text = text.lower()

    modality = None
    for non_xr, patterns in NON_XR_MODALITIES.items():
        if _contains_any(text, patterns):
            modality = non_xr
            break

    body_part = None
    for candidate, patterns in BODY_PART_PATTERNS.items():
        if _contains_any(text, patterns):
            body_part = candidate
            break

    if modality is None and (_contains_any(text, XR_PATTERNS) or body_part is not None):
        modality = "XR"

    laterality = "unknown"
    if re.search(r"\bbilateral\b", text):
        laterality = "bilateral"
    else:
        left = _contains_any(text, LEFT_PATTERNS)
        right = _contains_any(text, RIGHT_PATTERNS)
        if left and not right:
            laterality = "left"
        elif right and not left:
            laterality = "right"
        elif body_part in {"chest", "abdomen", "pelvis"}:
            laterality = "none"

    portable = None
    if _contains_any(text, [r"\bportable\b", r"\bbedside\b", r"\bmobile\b", r"portatil", r"portátil"]):
        portable = True
    elif body_part in {"chest", "abdomen", "pelvis", "knee", "hand", "ankle"}:
        portable = False

    pa = bool(re.search(r"\bpa\b", text))
    ap = bool(re.search(r"\bap\b", text))
    lat = _contains_any(text, [r"\blat\b", r"lateral"])
    projection_conflict = False
    if pa and lat:
        projection = "PA+LAT"
    elif ap and lat:
        projection = None
        projection_conflict = True
    elif ap and not pa:
        projection = "AP"
    elif pa and not ap:
        projection = "PA"
    elif _contains_any(text, [r"\boblique\b", r"\bobl\b"]):
        projection = "OBL"
    elif re.search(r"mortise", text):
        projection = "MORTISE"
    else:
        projection = None

    view_count, view_pattern = _detect_view_count(text, body_part)
    subtype = "KUB" if re.search(r"\bkub\b", text) else None

    return {
        "source_text": original,
        "modality": modality,
        "body_part": body_part,
        "laterality": laterality,
        "portable": portable,
        "projection": projection,
        "projection_conflict": projection_conflict,
        "view_count": view_count,
        "view_pattern": view_pattern,
        "subtype": subtype,
    }
