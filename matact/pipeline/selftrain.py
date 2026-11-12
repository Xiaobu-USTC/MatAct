"""Environment-grounded self-training (Supplementary Algorithm 1).

For every task in the pool, G candidate trajectories are rolled out in the current
environment, verified, scored with Q(tau), filtered with gamma_self and deduplicated
by tool sequence and answer similarity. The next training set is
D_cold U D_repair U Filter(D_self).
"""
from __future__ import annotations

from ..rollout import run_episode
from ..verify.scoring import score
from .quality import GAMMA_SELF, dedup_semantic, hard_failures


def rollout_candidates(tasks, make_policy, candidates=8, threshold=GAMMA_SELF, log=print, log_every=0):
    """Return (accepted trajectories, statistics). ``make_policy(task, k)`` builds the k-th sampler."""
    accepted, stats = [], {"rollouts": 0, "passed_verifiers": 0, "above_threshold": 0}
    for i, task in enumerate(tasks, start=1):
        for k in range(candidates):
            record, env = run_episode(task, make_policy(task, k))
            stats["rollouts"] += 1
            sc = score(task, record, env)
            if hard_failures(sc, True):
                continue
            stats["passed_verifiers"] += 1
            if sc["utility"] < threshold:
                continue
            stats["above_threshold"] += 1
            last = record["messages"][-1]["content"] if record["messages"] else ""
            accepted.append({
                "trajectory_id": f"self-{task['task_id']}-{k}", "task_id": task["task_id"], "source": "self",
                "steps": [s for s in record["steps"] if s.get("action")], "answer": record["answer"],
                "answer_rationale": last[:last.find("<answer>")].strip() if "<answer>" in last else "",
                "status": "answer", "repair_edges": [],
                "scores": {k2: sc[k2] for k2 in ("utility", "reward", "R_task", "R_exec", "R_sci", "R_ground", "C",
                                                 "cost", "success", "provenance_traceability")}})
        if log_every and i % log_every == 0:
            log(f"  {i}/{len(tasks)} tasks, {stats['above_threshold']} accepted")
    kept, dropped = dedup_semantic(accepted)
    stats.update(accepted_after_dedup=len(kept), removed_as_duplicates=len(dropped))
    return kept, stats


def next_training_set(cold, self_generated):
    """D_cold U D_repair U Filter(D_self); repair traces are part of the cold-start pool and kept explicitly."""
    repair = [t for t in cold if t.get("repair_edges")]
    seen, out = set(), []
    for t in list(cold) + repair + list(self_generated):
        if t["trajectory_id"] not in seen:
            seen.add(t["trajectory_id"])
            out.append(t)
    return out
