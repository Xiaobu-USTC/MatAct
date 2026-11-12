#!/usr/bin/env python3
"""Build MatAct-Bench, the seed trajectory pool and training-ready SFT / RL files.

    python scripts/build_dataset.py --out data            # full release (under a minute)
    python scripts/build_dataset.py --out /tmp/x --scale 0.05   # quick smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matact.agents.policies import ScriptedPolicy  # noqa: E402
from matact.env import materials  # noqa: E402
from matact.env.tools import TOOLS  # noqa: E402
from matact.pipeline.export import (LLAMAFACTORY_DATASET_INFO, to_messages, to_rl_row, to_sharegpt,  # noqa: E402
                                    verl_tool_config)
from matact.pipeline.seed_pool import build_seed_pool, summarize  # noqa: E402
from matact.pipeline.sources import SOURCE_RETAINED, SOURCE_SIZES  # noqa: E402
from matact.rollout import run_episode  # noqa: E402
from matact.tasks.generator import build_benchmark, suite_env_shift  # noqa: E402
from matact.verify.scoring import score  # noqa: E402


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def expert_demos(tasks):
    """Successful expert demonstrations (used as the validation set for SFT loss monitoring)."""
    out = []
    for task in tasks:
        record, env = run_episode(task, ScriptedPolicy())
        sc = score(task, record, env)
        if sc["success"]:
            last = record["messages"][-1]["content"]
            out.append(to_messages(task, {"trajectory_id": f"expert-{task['task_id']}", "source": "expert",
                                          "steps": record["steps"], "answer": record["answer"],
                                          "answer_rationale": last[:last.find("<answer>")].strip(),
                                          "scores": {"utility": sc["utility"]}}))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0, help="scale the seed-pool source sizes")
    ap.add_argument("--no-repair", action="store_true", help="drop the failure-repair source (ablation)")
    args = ap.parse_args()
    out, t0 = Path(args.out), time.time()

    splits, suites = build_benchmark(args.seed)
    for name, tasks in splits.items():
        write_jsonl(out / "benchmark" / f"{name}.jsonl", tasks)
    for name, tasks in suites.items():
        write_jsonl(out / "benchmark" / "suites" / f"{name}.jsonl", tasks)
    adaptation = suite_env_shift(args.seed, train=True)
    write_jsonl(out / "benchmark" / "env_shift_train.jsonl", adaptation)
    write_jsonl(out / "environment" / "materials.jsonl",
                (dict(m.record(), split=m.split, in_db_v1=m.in_db_v1, input_lattice_a_A=m.a) for m in materials.library()))
    write_json(out / "environment" / "tools.json",
               [{"name": t.name, "family": t.family, "cost": t.cost, "output": t.output, "typical_failure": t.failure,
                 "description": t.description, "parameters": t.parameters} for t in TOOLS])
    print(f"benchmark: {', '.join(f'{k}={len(v)}' for k, v in {**splits, **suites}.items())}")

    sizes = {k: max(1, round(v * args.scale)) for k, v in SOURCE_SIZES.items()}
    targets = {k: max(0, round(v * args.scale)) for k, v in SOURCE_RETAINED.items()}
    if args.no_repair:
        sizes["repair"] = 0
        targets["repair"] = 0
    protected = splits["valid"] + splits["test"] + [t for s in suites.values() for t in s]
    raws, retained, report, extra = build_seed_pool(splits["train"], protected, args.seed, sizes, targets)
    write_jsonl(out / "seed" / "raw_traces.jsonl", raws)
    write_jsonl(out / "seed" / "extra_tasks.jsonl", extra)
    write_jsonl(out / "seed" / "qc_report.jsonl", report)
    write_jsonl(out / "seed" / "retained_trajectories.jsonl", retained)
    summary = summarize(report)
    print("seed pool:", json.dumps(summary["sources"]))

    index = {t["task_id"]: t for t in splits["train"] + extra}
    messages = [to_messages(index[t["task_id"]], t) for t in retained]
    write_jsonl(out / "sft" / "train_messages.jsonl", messages)
    write_jsonl(out / "sft" / "sft_sharegpt.jsonl", (to_sharegpt(r) for r in messages))
    write_json(out / "sft" / "dataset_info.json", LLAMAFACTORY_DATASET_INFO)
    valid = expert_demos(splits["valid"])
    write_jsonl(out / "sft" / "valid_messages.jsonl", valid)
    write_jsonl(out / "rl" / "train.jsonl", (to_rl_row(t, i) for i, t in enumerate(splits["train"])))
    write_jsonl(out / "rl" / "valid.jsonl", (to_rl_row(t, i) for i, t in enumerate(splits["valid"])))
    write_json(out / "rl" / "verl_tools.yaml", verl_tool_config())

    chars = sorted(sum(len(m["content"]) for m in r["messages"]) for r in messages)
    stats = {
        "benchmark": {name: dict(Counter((t["category"], t["difficulty"]) for t in tasks).most_common())
                      for name, tasks in splits.items()},
        "benchmark_sizes": {**{k: len(v) for k, v in {**splits, **suites}.items()},
                            "env_shift_train": len(adaptation)},
        "seed_pool": summary,
        "sft": {"train_records": len(messages), "valid_records": len(valid),
                "by_source": dict(Counter(r["source"] for r in messages)),
                "by_category": dict(Counter(r["category"] for r in messages)),
                "message_chars": {"median": chars[len(chars) // 2], "p95": chars[int(0.95 * len(chars))],
                                  "max": chars[-1]} if chars else {}},
        "rl": {"train_prompts": len(splits["train"]), "valid_prompts": len(splits["valid"])},
        "seed": args.seed, "scale": args.scale,
    }
    stats["benchmark"] = {k: {f"{c}/{d}": n for (c, d), n in v.items()} for k, v in stats["benchmark"].items()}
    write_json(out / "stats.json", stats)
    print(f"sft: {len(messages)} training records, {len(valid)} validation records; rl: {len(splits['train'])} prompts")
    print(f"done in {time.time() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
