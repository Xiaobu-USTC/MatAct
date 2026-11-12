"""Assemble MatAct-Bench: in-distribution splits (T1-T4 x difficulty strata) and evaluation suites."""
from __future__ import annotations

import copy
import random

from ..env import materials
from . import builders_t12 as b12
from . import builders_t34 as b34

STRATA = (("regular", 0.50), ("missing_data", 0.20), ("tool_failure", 0.15), ("long_horizon", 0.15))
SPLIT_SIZES = {"train": 900, "valid": 100, "test": 100}
SPLIT_CODE = {"train": "TR", "valid": "VA", "test": "TE"}
CATEGORY_PREFIX = {"structure": "T1", "first_principles": "T2", "ml_potential": "T3", "database": "T4"}


def strata_slots(n):
    counts = [round(n * r) for _, r in STRATA]
    counts[0] += n - sum(counts)
    return [d for (d, _), c in zip(STRATA, counts) for _ in range(c)]


def _pick(rng, pool, pred=lambda m: True):
    options = [m for m in pool if pred(m)]
    if not options:
        raise ValueError("no material satisfies the constraint")
    return rng.choice(options)


def _two(rng, pool, pred=lambda m: True, max_mismatch=None):
    first = _pick(rng, pool, pred)

    def second(m):
        close = max_mismatch is None or abs(m.a_ref - first.a_ref) / first.a_ref <= max_mismatch
        return pred(m) and m.formula != first.formula and close
    try:
        return first, _pick(rng, pool, second)
    except ValueError:
        return first, _pick(rng, pool, lambda m: pred(m) and m.formula != first.formula)


def _cubic_insulator(m):
    return not m.metallic and m.prototype != "wurtzite"


def build(category, difficulty, rng, pool, idx, filters):
    if category == "structure":
        if difficulty == "regular":
            return b12.t1_retrieve(rng, _pick(rng, pool)) if rng.random() < 0.6 else b12.t1_file(rng, _pick(rng, pool), idx)
        if difficulty == "missing_data":
            while True:
                m = _pick(rng, pool)
                proto = b12.missing_prototype(rng, m)
                if proto:
                    return b12.t1_retrieve(rng, m, prototype=proto)
        if difficulty == "tool_failure":
            return b12.t1_file(rng, _pick(rng, pool), idx, mode="repair")
        return b12.t1_density(rng, *_two(rng, pool))
    if category == "first_principles":
        if difficulty == "regular":
            return b12.t2_property(rng, _pick(rng, pool)) if rng.random() < 0.75 else b12.t2_singlepoint(rng, _pick(rng, pool), idx)
        if difficulty == "missing_data":
            while True:
                m = _pick(rng, pool)
                proto = b12.missing_prototype(rng, m)
                if proto:
                    return b12.t2_property(rng, m, prototype=proto)
        if difficulty == "tool_failure":
            return b12.t2_property(rng, _pick(rng, pool), fault="NON_CONVERGENCE")
        return b12.t2_mismatch(rng, *_two(rng, pool, _cubic_insulator, max_mismatch=0.08))
    if category == "ml_potential":
        if difficulty == "regular":
            return b34.t3_finetune(rng, _pick(rng, pool), idx) if rng.random() < 0.6 else b34.t3_pretrained(rng, _pick(rng, pool), idx)
        if difficulty == "missing_data":
            other = _pick(rng, pool) if rng.random() < 0.5 else None
            return b34.t3_finetune(rng, _pick(rng, pool), idx, missing=True, other=other)
        if difficulty == "tool_failure":
            return b34.t3_finetune(rng, _pick(rng, pool), idx, fault=True)
        return b34.t3_finetune(rng, _pick(rng, pool), idx, compare=True)
    if difficulty == "regular":
        r = rng.random()
        if r < 0.5:
            return b34.t4_lookup(rng, _pick(rng, pool, lambda m: m.in_db_v1))
        if r < 0.75 and filters:
            return b34.t4_filter(rng, filters)
        return b34.t4_insert(rng, [_pick(rng, pool)], idx)
    if difficulty == "missing_data":
        return b34.t4_lookup(rng, _pick(rng, pool, lambda m: not m.in_db_v1))
    if difficulty == "tool_failure":
        return b34.t4_insert(rng, [_pick(rng, pool)], idx, fault=True)
    return b34.t4_insert(rng, list(_two(rng, pool)), idx)


def _finish(task, task_id, suite, split):
    return {"task_id": task_id, "suite": suite, "split": split, **task}


def materials_in(*splits):
    return [m for m in materials.library() if m.split in splits]


def id_split(split, seed=0):
    pool = materials_in(split)
    filters = b34.filter_candidates(("train",) if split == "train" else (split, "train"))
    tasks = []
    for category, prefix in CATEGORY_PREFIX.items():
        rng = random.Random(f"{seed}-{split}-{category}")
        slots = strata_slots(SPLIT_SIZES[split])
        rng.shuffle(slots)
        for i, difficulty in enumerate(slots):
            task = build(category, difficulty, rng, pool, i, filters)
            tasks.append(_finish(task, f"{prefix}-{SPLIT_CODE[split]}-{i:04d}", "id", split))
    return tasks


def suite_ood_chem(seed=0):
    pool = materials_in("ood")
    filters = b34.filter_candidates(("ood", "train", "valid", "test"))
    filters = [f for f in filters if any(materials.by_id(i).split == "ood" for i in f[4])]
    out = []
    for c, category in enumerate(CATEGORY_PREFIX):
        rng = random.Random(f"{seed}-oodchem-{category}")
        for i in range(25):
            out.append(_finish(build(category, "regular", rng, pool, i, filters), f"OODCHEM-{c * 25 + i:04d}", "ood_chem", "eval"))
    return out


def suite_ood_comp(seed=0):
    rng = random.Random(f"{seed}-oodcomp")
    test, gaps = materials_in("test"), [m for m in materials_in("test", "valid") if not m.in_db_v1]
    out = [b34.c_compute_register(rng, _pick(rng, test)) for _ in range(50)]
    out += [b34.c_db_fallback(rng, _pick(rng, gaps)) for _ in range(50)]
    return [_finish(t, f"OODCOMP-{i:04d}", "ood_comp", "eval") for i, t in enumerate(out)]


def suite_paraphrase(test_tasks):
    out = []
    for task in test_tasks:
        for k, style in enumerate(("terse", "novice", "mixed"), start=1):
            t = copy.deepcopy(task)
            t.update(task_id=f"{task['task_id']}-P{k}", suite="paraphrase", split="eval", query=task["query_variants"][style])
            t["params"]["style"] = style
            out.append(t)
    return out


def suite_tool_failure(seed=0):
    rng = random.Random(f"{seed}-toolfail")
    test = materials_in("test")
    out = [b12.t2_property(rng, _pick(rng, test), fault="TIMEOUT") for _ in range(25)]
    out += [b12.t2_property(rng, _pick(rng, test), fault="NON_CONVERGENCE") for _ in range(25)]
    out += [b34.t4_lookup(rng, _pick(rng, test, lambda m: not m.in_db_v1)) for _ in range(25)]
    out += [b12.t1_file(rng, _pick(rng, test), i, mode="wrong_path") for i in range(25)]
    return [_finish(t, f"TOOLFAIL-{i:04d}", "tool_failure", "eval") for i, t in enumerate(out)]


def suite_env_shift(seed=0, train=False):
    """Tasks in the updated environment v2. The evaluation suite uses test materials; ``train=True``
    gives the adaptation pool (training materials) for the extra self-training round."""
    rng = random.Random(f"{seed}-envshift" + ("-train" if train else ""))
    pool = materials_in("train") if train else materials_in("test")
    gaps = [m for m in (pool if train else materials_in("test", "valid")) if not m.in_db_v1]
    n = 4 if train else 1
    out = [b34.t4_lookup(rng, _pick(rng, gaps), env_version="v2") for _ in range(40 * n)]
    out += [b34.t3_pretrained(rng, _pick(rng, pool), i, env_version="v2") for i in range(30 * n)]
    out += [b12.t2_property(rng, _pick(rng, pool), env_version="v2") for _ in range(30 * n)]
    prefix, split = ("ENVSHIFT-TR", "train") if train else ("ENVSHIFT", "eval")
    return [_finish(t, f"{prefix}-{i:04d}", "env_shift", split) for i, t in enumerate(out)]


def suite_stress(seed=0):
    rng = random.Random(f"{seed}-stress")
    test, ood = materials_in("test"), materials_in("ood")
    filters = b34.filter_candidates(("test", "train"))
    out = []
    for i in range(50):
        m = _pick(rng, ood)
        out.append(("ood_structures", b12.t1_retrieve(rng, m) if i % 2 else b12.t2_property(rng, m)))
    for condition, difficulty in (("tool_failures", "tool_failure"), ("missing_data", "missing_data")):
        for i in range(50):
            category = list(CATEGORY_PREFIX)[i % 4]
            out.append((condition, build(category, difficulty, rng, test, i, filters)))
    for _ in range(50):
        out.append(("high_cost", b12.t2_property(rng, _pick(rng, test), budget_factor=1.25)))
    tasks = []
    for i, (condition, t) in enumerate(out):
        t["params"]["stress_condition"] = condition
        tasks.append(_finish(t, f"STRESS-{i:04d}", "stress", "eval"))
    return tasks


def build_benchmark(seed=0):
    splits = {s: id_split(s, seed) for s in SPLIT_SIZES}
    suites = {"ood_chem": suite_ood_chem(seed), "ood_comp": suite_ood_comp(seed),
              "paraphrase": suite_paraphrase(splits["test"]), "tool_failure": suite_tool_failure(seed),
              "env_shift": suite_env_shift(seed), "stress": suite_stress(seed)}
    return splits, suites
