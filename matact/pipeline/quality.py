"""Quality control: replay verification, hard filters, utility threshold, deduplication and leakage control."""
from __future__ import annotations

import json
import re

from ..rollout import replay
from ..verify.scoring import score

GAMMA_COLD = 0.72
GAMMA_SELF = 0.78


def hard_failures(sc: dict, matched: bool) -> list:
    reasons = []
    if not matched:
        reasons.append("unverifiable observations")
    if sc["parse_errors"]:
        reasons.append("malformed calls")
    if sc["grounding"]["unknown_citations"]:
        reasons.append("unverifiable artifacts")
    if sc["provenance_traceability"] < 1.0:
        reasons.append("unsupported conclusion")
    if not sc["success"]:
        reasons.append("missing completion state")
    return reasons


def assess(task: dict, traj: dict):
    """Replay the trajectory, score it and return (score, hard-filter reasons)."""
    env, matched = replay(task, traj["steps"])
    sc = score(task, traj, env)
    return sc, hard_failures(sc, matched)


def _key(traj: dict) -> str:
    return json.dumps([traj["task_id"], [s.get("action") for s in traj["steps"]], traj["answer"]], sort_keys=True)


def _rank(traj: dict):
    sc = traj["scores"]
    return (-sc["utility"], sc["cost"], len(traj["steps"]))


def dedup_exact(trajs: list):
    """Among identical (task, actions, answer) traces keep the highest utility, then lower cost, then shorter."""
    best = {}
    for t in trajs:
        k = _key(t)
        if k not in best or _rank(t) < _rank(best[k]):
            best[k] = t
    kept = {id(t) for t in best.values()}
    return [t for t in trajs if id(t) in kept], [t["trajectory_id"] for t in trajs if id(t) not in kept]


def grams(text: str, n: int = 3) -> set:
    text = re.sub(r"\s+", " ", text.lower())
    return {text[i:i + n] for i in range(max(1, len(text) - n + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def dedup_semantic(trajs: list, threshold: float = 0.9):
    """Self-training deduplication by tool sequence and answer similarity (per task)."""
    kept, dropped = [], []
    for t in sorted(trajs, key=_rank):
        seq = [(s.get("action") or {}).get("name") for s in t["steps"]]
        g = grams(t["answer"])
        if any(k["task_id"] == t["task_id"] and k["_seq"] == seq and jaccard(k["_grams"], g) >= threshold for k in kept):
            dropped.append(t["trajectory_id"])
            continue
        t["_seq"], t["_grams"] = seq, g
        kept.append(t)
    for t in kept:
        t.pop("_seq"), t.pop("_grams")
    return kept, dropped


def task_materials(task: dict) -> set:
    ref = task["reference"]
    ids = set(ref.get("material_ids", [])) | set(ref.get("ids", []))
    ids |= {ref["material_id"]} if "material_id" in ref else set()
    ids |= {r["material_id"] for r in ref.get("records", [])}
    ids |= {s["material_id"] for s in task["initial_state"]["workspace"].values() if "material_id" in s}
    return ids


def _signature(task: dict) -> str:
    return json.dumps([task["subtype"], task["params"]], sort_keys=True)


class LeakageGuard:
    """Reject synthetic tasks that overlap validation/test tasks at the structure (material), query
    (identical normalized request) or semantic (same workflow and parameters) level."""

    def __init__(self, protected_tasks):
        from ..env import materials
        self.materials, self.queries, self.signatures = set(), set(), set()
        for t in protected_tasks:
            self.materials |= {m for m in task_materials(t) if materials.by_id(m).split != "train"}
            self.queries.add(re.sub(r"\s+", " ", t["query"].lower()).strip())
            self.signatures.add(_signature(t))

    def __call__(self, task: dict):
        if task_materials(task) & self.materials:
            return "overlaps validation/test materials"
        if re.sub(r"\s+", " ", task["query"].lower()).strip() in self.queries:
            return "duplicates a validation/test query"
        if _signature(task) in self.signatures:
            return "semantically identical to a validation/test task"
        return None
