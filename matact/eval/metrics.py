"""Aggregate metrics (SR, FR, FCR, ESR, ATC, CPS) and the evaluation harness (Supplementary Algorithm 2)."""
from __future__ import annotations

from collections import defaultdict

from ..rollout import run_episode
from ..verify.scoring import score

TIER = {"db_value": "A", "dft_value": "B", "dft_mismatch": "B", "dft_write": "B"}


def evaluate_policy(tasks, make_policy, validator=None, log_every=0, log=print):
    """Run ``make_policy(task)`` on every task; returns per-task rows (scores plus trajectory)."""
    rows = []
    for i, task in enumerate(tasks, start=1):
        record, env = run_episode(task, make_policy(task), validator=validator)
        row = score(task, record, env)
        row.update(suite=task["suite"], category=task["category"], subtype=task["subtype"],
                   difficulty=task["difficulty"], tier=TIER.get(task["reference"]["check"]),
                   status=record["status"], trajectory=record)
        rows.append(row)
        if log_every and i % log_every == 0:
            log(f"  {i}/{len(tasks)} tasks")
    return rows


def aggregate(rows, reference_cps=None) -> dict:
    """Metrics in percent (SR, FR, FCR, ESR), mean calls (ATC) and cost per successful task (CPS)."""
    n = len(rows)
    if not n:
        return {}
    calls = sum(r["n_calls"] for r in rows)
    success = sum(r["success"] for r in rows)
    cost = sum(r["cost"] for r in rows)
    cps = cost / (success + 1e-9)
    out = {"n_tasks": n,
           "SR": round(100 * success / n, 2),
           "FR": round(100 * sum(r["factual"] for r in rows) / n, 2),
           "FCR": round(100 * sum(r["n_format_ok"] for r in rows) / max(1, sum(r["n_turns"] for r in rows)), 2),
           "ESR": round(100 * sum(r["n_exec_ok"] for r in rows) / max(1, calls), 2),
           "ATC": round(calls / n, 2),
           "CPS_units": round(cps, 3),
           "PT": round(100 * sum(r["provenance_traceability"] for r in rows) / n, 2)}
    if reference_cps:
        out["CPS"] = round(cps / reference_cps, 3)
    return out


def breakdown(rows, key: str) -> dict:
    groups = defaultdict(list)
    for r in rows:
        if r.get(key) is not None:
            groups[r[key]].append(r)
    return {k: aggregate(v) for k, v in sorted(groups.items())}


def report(rows, reference_cps=None) -> dict:
    return {"overall": aggregate(rows, reference_cps), "by_category": breakdown(rows, "category"),
            "by_difficulty": breakdown(rows, "difficulty"), "by_tier": breakdown(rows, "tier")}
