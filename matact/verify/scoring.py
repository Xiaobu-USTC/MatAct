"""Trajectory scoring: verifier labels, utility Q(tau), verifiable reward R(tau) and per-task metrics.

Q(tau) = 0.30 R_task + 0.25 R_exec + 0.20 R_sci + 0.15 R_ground - 0.10 C(tau)      (Supplementary Eq. S2)
R(tau) = 0.40 R_ts   + 0.25 R_ex   + 0.25 R_gf  - 0.10 C(tau), clipped to [-1, 1] (Supplementary Eq. S3)
"""
from __future__ import annotations

from ..env.common import ERROR_LABELS, in_workspace
from ..env.tools import COST, REGISTRY, check_arguments
from .answer import grounding
from .checks import check_task

FAILURE_LABELS = ("ARGUMENT", "UNAVAILABLE", "PRECONDITION", "EXECUTION", "ARTIFACT", "COST")
_REFERENCE = {}


def reference_run(task: dict) -> dict:
    """Cost and unavoidable error codes of the scripted expert on ``task`` (cached)."""
    if task["task_id"] not in _REFERENCE:
        from ..agents.policies import ScriptedPolicy
        from ..rollout import run_episode
        record, _ = run_episode(task, ScriptedPolicy())
        executed = [s for s in record["steps"] if s.get("action")]
        _REFERENCE[task["task_id"]] = {
            "cost": sum(COST.get(s["action"]["name"], 0) for s in executed),
            "errors": sorted({s["observation"].get("code") for s in executed if s["observation"]["status"] == "error"}),
        }
    return _REFERENCE[task["task_id"]]


def label(step) -> str:
    if step.get("parse_error") is not None:
        return "PARSE"
    obs = step["observation"]
    if obs.get("status") == "success":
        return ""
    if obs.get("status") == "rejected":
        return "VALIDATOR"
    return ERROR_LABELS.get(obs.get("code"), "EXECUTION")


def _same_family(a: str, b: str) -> bool:
    return a == b or REGISTRY[a].family == REGISTRY[b].family if a in REGISTRY and b in REGISTRY else False


def format_ok(step) -> bool:
    """Formatting compliance of one policy turn: parsable call, registered tool name, schema-valid
    arguments and workspace paths. Uses the action as proposed by the policy, before any validator repair."""
    if step.get("parse_error") is not None:
        return False
    action = step.get("proposed_action") or step.get("action")
    if not action or action.get("name") not in REGISTRY:
        return False
    args = action.get("arguments")
    if check_arguments(action["name"], args):
        return False
    return all(in_workspace(v) for k, v in args.items() if isinstance(v, str) and k.endswith("uri"))


def score(task: dict, record: dict, env) -> dict:
    steps, answer = record["steps"], record.get("answer")
    ref = reference_run(task)
    verdict = check_task(task, steps, answer, env)
    checks = verdict["checks"]
    r_ground, pt, gdet = grounding(task, steps, answer, env)
    labels = [label(s) for s in steps]
    turns = [s for s in steps if s.get("action") or s.get("parse_error") is not None or s.get("proposed_action")]
    executed = [i for i, s in enumerate(steps) if s.get("action")]
    exec_ok, retries, redundant, seen = [], 0, 0, []
    for i in executed:
        s, lab = steps[i], labels[i]
        key = (s["action"]["name"], repr(sorted(s["action"]["arguments"].items())))
        if lab == "" and key in seen:
            redundant += 1
        if lab == "":
            seen.append(key)
        if lab in ("", "NO_DATA"):
            exec_ok.append(1.0)
            continue
        retries += 1
        expected = s["observation"].get("code") in ref["errors"]
        repaired = any(labels[j] == "" and _same_family(steps[j]["action"]["name"], s["action"]["name"])
                       for j in executed if j > i)
        exec_ok.append(1.0 if expected and (repaired or task["difficulty"] == "missing_data") else 0.0)
    parse_errors = sum(1 for lab in labels if lab == "PARSE")
    n_turns = max(1, len(executed) + parse_errors)
    r_exec = (sum(exec_ok)) / n_turns if executed else 0.0
    n_format_ok = sum(1 for s in turns if format_ok(s))
    exec_success = sum(1 for i in executed if labels[i] not in FAILURE_LABELS)
    cost = sum(COST.get(steps[i]["action"]["name"], 0) for i in executed)
    excess = max(0.0, cost / ref["cost"] - 1.0) if ref["cost"] else (1.0 if cost else 0.0)
    n = max(1, len(executed))
    failures = retries - sum(1 for i in executed if steps[i]["observation"].get("code") in ref["errors"])
    c_tau = min(1.0, 0.6 * min(1.0, excess) + 0.2 * min(1.0, 3 * redundant / n) + 0.2 * min(1.0, 2 * failures / n))
    values = list(checks.values())
    r_task = sum(values) / len(values) if values else 0.0
    r_sci = sum(verdict["science"]) / len(verdict["science"]) if verdict["science"] else 1.0
    completed = answer is not None and record.get("status") == "answer"
    success = completed and all(values)
    if not completed:
        r_task = r_ground = 0.0
    utility = 0.30 * r_task + 0.25 * r_exec + 0.20 * r_sci + 0.15 * r_ground - 0.10 * c_tau
    format_rate = n_format_ok / max(1, len(turns))
    reward = max(-1.0, min(1.0, 0.40 * (0.6 * r_task + 0.4 * r_sci) + 0.25 * r_exec +
                           0.25 * (0.5 * format_rate + 0.5 * r_ground) - 0.10 * c_tau))
    correct = completed and all(checks[k] for k in verdict["value_checks"])
    factual = completed and (correct or (pt == 1.0 and not gdet["unknown_citations"]))
    return {
        "task_id": task["task_id"], "success": success, "factual": factual, "checks": checks,
        "answer_correct": correct,
        "n_calls": len(executed), "n_format_ok": n_format_ok, "n_exec_ok": exec_success,
        "n_turns": len(turns), "cost": cost, "reference_cost": ref["cost"], "redundant_calls": redundant,
        "failed_calls": retries, "parse_errors": parse_errors, "labels": [x for x in labels if x],
        "R_task": round(r_task, 4), "R_exec": round(r_exec, 4), "R_sci": round(r_sci, 4), "R_ground": round(r_ground, 4),
        "C": round(c_tau, 4), "utility": round(utility, 4), "reward": round(reward, 4),
        "provenance_traceability": round(pt, 4), "grounding": gdet,
    }
