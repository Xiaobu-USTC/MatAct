#!/usr/bin/env python3
"""Compare two methods on per-task outcomes (Methods, Statistics).

Per-task outcomes are averaged over runs (seeds); differences are tested with two-sided paired
permutation tests (10,000 permutations) with Holm-Bonferroni correction across suites, and 95%
confidence intervals come from a task-level bootstrap (10,000 resamples).

    python scripts/compare.py \
        --a results/matact_s13.trajectories.jsonl results/matact_s42.trajectories.jsonl \
        --b results/sft_s13.trajectories.jsonl results/sft_s42.trajectories.jsonl --metric success
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matact.eval.stats import bootstrap_ci, holm_bonferroni, paired_permutation_test  # noqa: E402


def load(files, metric):
    """suite -> task_id -> outcome averaged over the given result files (runs)."""
    values = defaultdict(lambda: defaultdict(list))
    for name in files:
        with open(name, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                values[row["suite"]][row["task_id"]].append(float(row[metric]))
    return {s: {t: sum(v) / len(v) for t, v in tasks.items()} for s, tasks in values.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", nargs="+", required=True, help="trajectory files of method A (one per run)")
    ap.add_argument("--b", nargs="+", required=True, help="trajectory files of method B (one per run)")
    ap.add_argument("--metric", default="success", choices=["success", "factual", "answer_correct"])
    ap.add_argument("--permutations", type=int, default=10000)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    a, b = load(args.a, args.metric), load(args.b, args.metric)
    rows, p_values = {}, {}
    for suite in sorted(set(a) & set(b)):
        ids = sorted(set(a[suite]) & set(b[suite]))
        xa, xb = [a[suite][t] for t in ids], [b[suite][t] for t in ids]
        diff = [x - y for x, y in zip(xa, xb)]
        lo, hi = bootstrap_ci(diff, args.resamples, seed=args.seed)
        p_values[suite] = paired_permutation_test(xa, xb, args.permutations, args.seed)
        rows[suite] = {"n_tasks": len(ids), "A_%": round(100 * sum(xa) / len(ids), 2),
                       "B_%": round(100 * sum(xb) / len(ids), 2), "diff_pp": round(100 * sum(diff) / len(ids), 2),
                       "ci95_pp": [round(100 * lo, 2), round(100 * hi, 2)], "p": round(p_values[suite], 5)}
    for suite, p in holm_bonferroni(p_values).items():
        rows[suite]["p_holm"] = round(p, 5)
    print(json.dumps({"metric": args.metric, "suites": rows}, indent=2))


if __name__ == "__main__":
    main()
