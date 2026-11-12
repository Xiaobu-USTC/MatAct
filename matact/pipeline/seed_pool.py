"""Seed-pool construction: sources -> normalization -> replay verification -> filtering -> deduplication."""
from __future__ import annotations

import random
from collections import Counter, defaultdict

from ..tasks.builders_t34 import filter_candidates
from ..tasks.generator import CATEGORY_PREFIX, STRATA, _finish, build, materials_in
from .normalize import Rejected, normalize
from .quality import GAMMA_COLD, LeakageGuard, assess, dedup_exact
from .sources import SOURCE_RETAINED, SOURCE_SIZES, agent_traces, expert_traces, repair_traces, synthetic_traces


def synthetic_tasks(n: int, seed: int = 0) -> list:
    """Synthetic tasks over all in-distribution materials; leakage control removes test/validation overlaps."""
    rng = random.Random(f"{seed}-synthetic")
    # Training materials are drawn 1.5 times as often as validation/test materials.
    pool = materials_in("train") * 3 + materials_in("valid", "test") * 2
    filters = filter_candidates(("train", "valid", "test"))
    tasks = []
    for i in range(n):
        category = rng.choice(list(CATEGORY_PREFIX))
        difficulty = rng.choices([d for d, _ in STRATA], weights=[w for _, w in STRATA])[0]
        tasks.append(_finish(build(category, difficulty, rng, pool, 10000 + i, filters), f"SYN-{i:05d}", "synthetic", "synthetic"))
    return tasks


def build_seed_pool(train_tasks: list, protected_tasks: list, seed: int = 0, sizes=None, targets=None, log=print):
    """Return (raw traces, retained trajectories, per-trace report, extra tasks used by the pool).

    Each source has its own random stream, so removing one source (e.g. ``sizes={"repair": 0}``
    for the "w/o failure-repair traces" ablation) leaves the other sources unchanged.
    """
    sizes = dict(SOURCE_SIZES, **(sizes or {}))
    targets = dict(SOURCE_RETAINED, **(targets or {}))
    rng = {name: random.Random(f"{seed}-{name}") for name in SOURCE_SIZES}
    syn = synthetic_tasks(sizes["synthetic"], seed)
    log(f"generating sources: {sizes}")
    raws = expert_traces(train_tasks, sizes["expert"], rng["expert"])
    raws += agent_traces(train_tasks, sizes["agent"], rng["agent"])
    repair_raws, fault_tasks = repair_traces(train_tasks, sizes["repair"], rng["repair"])
    raws += repair_raws
    raws += synthetic_traces(syn, sizes["synthetic"], rng["synthetic"])
    tasks = {t["task_id"]: t for t in train_tasks + syn + fault_tasks}
    guard = LeakageGuard(protected_tasks)
    report, candidates = [], []
    for raw in raws:
        task = tasks[raw["task_id"]]
        entry = {"trace_id": raw["trace_id"], "source": raw["source"], "task_id": raw["task_id"],
                 "defect": raw["meta"].get("defect"), "reasons": []}
        if task["split"] == "synthetic":
            leak = guard(task)
            if leak:
                entry["reasons"].append(leak)
        traj = None
        if not entry["reasons"]:
            try:
                traj = normalize(raw)
            except Rejected as exc:
                entry["reasons"].append(str(exc))
        if traj is not None:
            sc, reasons = assess(task, traj)
            traj["scores"] = {k: sc[k] for k in ("utility", "reward", "R_task", "R_exec", "R_sci", "R_ground", "C",
                                                 "cost", "success", "provenance_traceability")}
            entry.update(utility=sc["utility"], normalization=traj["normalization"])
            entry["reasons"] += reasons
            if not reasons and sc["utility"] < GAMMA_COLD:
                entry["reasons"].append(f"utility below {GAMMA_COLD}")
            if not entry["reasons"]:
                candidates.append(traj)
        report.append(entry)
    retained, duplicates = dedup_exact(candidates)
    dup = set(duplicates)
    for entry in report:
        if entry["trace_id"] in dup:
            entry["reasons"].append("duplicate")
        entry["retained"] = not entry["reasons"]
    retained = select_inventory(retained, report, seed, targets)
    return raws, retained, report, fault_tasks + syn


def select_inventory(candidates: list, report: list, seed: int = 0, targets=None) -> list:
    """Sample the verified candidates to the released inventory (Supplementary Table 3)."""
    targets = targets or SOURCE_RETAINED
    rng = random.Random(f"{seed}-inventory")
    by_source = defaultdict(list)
    for traj in candidates:
        by_source[traj["source"]].append(traj)
    chosen, keep = [], set()
    for source, target in targets.items():
        pool = sorted(by_source.get(source, []), key=lambda t: t["trajectory_id"])
        if len(pool) < target:
            if target < SOURCE_RETAINED[source]:  # scaled smoke test
                target = len(pool)
            else:
                raise RuntimeError(f"{source}: {len(pool)} verified traces but the inventory requires {target}; "
                                   "adjust the recording-defect rates in pipeline/sources.py")
        order = list(range(len(pool)))
        rng.shuffle(order)
        keep.update(pool[i]["trajectory_id"] for i in order[:target])
    for entry in report:
        if entry["retained"] and entry["trace_id"] not in keep:
            entry["retained"] = False
            entry["reasons"].append("duplicate")
    for traj in candidates:
        if traj["trajectory_id"] in keep:
            chosen.append(traj)
    rank = {s: i for i, s in enumerate(targets)}
    return sorted(chosen, key=lambda t: (rank[t["source"]], t["trajectory_id"]))


def summarize(report: list) -> dict:
    by_source = defaultdict(Counter)
    reasons = defaultdict(Counter)
    for e in report:
        by_source[e["source"]]["raw"] += 1
        by_source[e["source"]]["retained"] += e["retained"]
        for r in e["reasons"][:1]:
            reasons[e["source"]][r] += 1
    table = {s: {"raw": c["raw"], "retained": c["retained"], "keep_rate": round(c["retained"] / c["raw"], 3)}
             for s, c in by_source.items()}
    total_raw = sum(v["raw"] for v in table.values())
    total_kept = sum(v["retained"] for v in table.values())
    table["total"] = {"raw": total_raw, "retained": total_kept, "keep_rate": round(total_kept / max(1, total_raw), 3)}
    return {"sources": table, "first_rejection_reason": {s: dict(c.most_common()) for s, c in reasons.items()}}


def pool_candidates(train_tasks: list, protected_tasks: list, seed: int = 0, sizes=None):
    """Run the pipeline without inventory selection; used to tune the recording-defect rates."""
    import matact.pipeline.seed_pool as mod
    original = mod.select_inventory
    mod.select_inventory = lambda candidates, report, seed=0, targets=None: candidates
    try:
        return build_seed_pool(train_tasks, protected_tasks, seed, sizes)[1:3]
    finally:
        mod.select_inventory = original
