"""Answer analysis: numeric claims with units, citations and provenance traceability."""
from __future__ import annotations

import json
import re

_URI = re.compile(r"workspace://[^\s,;()\"'`]+")
_OBS = re.compile(r"\bobs-\d+\b")
_MASKS = [
    _URI,
    re.compile(r"\b(?:obs|mx|tx|q)-\d+\b"),
    re.compile(r"\b[a-z][a-z0-9]*(?:-[a-z0-9.]+)+\b"),      # model and code names, e.g. qe-7.2, chgnet-0.3
    re.compile(r"\b\d+\s*[x×]\s*\d+\s*[x×]\s*\d+\b"),       # supercells and k-meshes
    re.compile(r"\b(?:[A-Z][a-z]?\d*){1,5}\b"),              # chemical formulas
    re.compile(r"\b[Tt]ier-?[ABC]\b|\bT[1-4]\b"),
]
_NUMBER = re.compile(r"(?<![\w.])([-+−]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*(meV/atom|eV/atom|meV/Å|meV/A|eV/Å|g/cm³|g/cm3|"
                     r"meV|eV|Å|%|kbar|h)?(?![\w/])")
SCALES = (1.0, 1e3, 1e-3, 100.0, 0.01, 13.605693122994, 1 / 13.605693122994)


def _clean(uri: str) -> str:
    return uri.rstrip(".:;,!?")


def citations(answer: str):
    uris = {_clean(u) for u in _URI.findall(answer or "")}
    return uris, set(_OBS.findall(answer or ""))


def numbers(answer: str):
    """Numeric claims as (value, unit, decimals); identifiers, formulas and URIs are ignored."""
    text = answer or ""
    for pattern in _MASKS:
        text = pattern.sub(" ", text)
    out = []
    for raw, unit in _NUMBER.findall(text):
        raw = raw.replace("−", "-")
        decimals = len(raw.split(".")[1].split("e")[0].split("E")[0]) if "." in raw else 0
        out.append((float(raw), unit or "", decimals))
    return out


def _collect(value, sink):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        sink.append(float(value))
    elif isinstance(value, dict):
        for v in value.values():
            _collect(v, sink)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _collect(v, sink)
    elif isinstance(value, str) and not value.startswith("workspace://"):
        sink.extend(float(x) for x in re.findall(r"(?<![\w.-])-?\d+(?:\.\d+)?(?![\w.])", value))


def evidence(task: dict, steps: list):
    """Numbers available as evidence: query, issued arguments and observations.

    Returns (all numbers, primary numbers) where primary numbers come from result-bearing
    observations and are used for derived quantities (differences, ratios, percentages).
    """
    every, primary = [], []
    _collect([float(x) for x in re.findall(r"(?<![\w.-])-?\d+(?:\.\d+)?(?![\w.])", task.get("query", ""))], every)
    for step in steps:
        if step.get("action"):
            _collect(step["action"].get("arguments", {}), every)
        obs = step.get("observation", {})
        _collect({k: v for k, v in obs.items() if k != "observation_id"}, every)
        if obs.get("status") == "success" and (step.get("action") or {}).get("name") in (
                "result.extract", "mlp.evaluate", "database.query", "structure.validate"):
            _collect({k: v for k, v in obs.items() if k != "observation_id"}, primary)
    return every, sorted(set(primary))[:80]


def _close(x, y, decimals):
    tol = 0.51 * 10 ** (-decimals) + 1e-9 + 1e-6 * abs(y)
    return abs(x - y) <= tol


def traceable(value, decimals, every, primary) -> bool:
    for o in every:
        for s in SCALES:
            if _close(value, o * s, decimals):
                return True
    for a in primary:
        for b in primary:
            if a == b:
                continue
            for derived in (a - b, (a - b) * 1e3):
                if _close(value, derived, decimals) or _close(value, abs(derived), decimals):
                    return True
            if b and _close(value, (a - b) / b * 100, decimals):
                return True
    return False


def provenance(task: dict, steps: list, answer: str):
    """Fraction of numeric claims traceable to evidence (PT) and the untraceable values."""
    claims = numbers(answer)
    if not claims:
        return 1.0, []
    every, primary = evidence(task, steps)
    missing = [v for v, _, d in claims if not traceable(v, d, every, primary)]
    return 1.0 - len(missing) / len(claims), missing


def known_uris(steps: list, env=None):
    found = set()
    for step in steps:
        found.update(_URI.findall(json.dumps(step.get("observation", {}), ensure_ascii=False)))
    if env is not None:
        found.update(env.artifacts)
    return {_clean(u) for u in found}


def grounding(task, steps, answer, env=None):
    """Return (R_ground in [0, 1], PT, details)."""
    if not answer:
        return 0.0, 0.0, {"cited": False, "unknown_citations": [], "untraceable": []}
    uris, obs_ids = citations(answer)
    valid_obs = {s["observation"]["observation_id"] for s in steps if "observation" in s}
    known = known_uris(steps, env)
    unknown = sorted(u for u in uris if u not in known) + sorted(o for o in obs_ids if o not in valid_obs)
    cited = bool((uris & known) or (obs_ids & valid_obs))
    pt, missing = provenance(task, steps, answer)
    score = (0.5 * cited + 0.5 * pt) if not unknown else 0.0
    return score, pt, {"cited": cited, "unknown_citations": unknown, "untraceable": missing}
