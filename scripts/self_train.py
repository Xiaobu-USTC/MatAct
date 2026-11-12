#!/usr/bin/env python3
"""One round of environment-grounded self-training data selection (Supplementary Algorithm 1).

Rolls out G candidates per training task with the current policy, keeps trajectories that
pass the verifiers with Q(tau) >= 0.78, deduplicates them and writes the next SFT set
D_cold U D_repair U Filter(D_self) in the same formats as scripts/build_dataset.py.

    python scripts/self_train.py --policy openai --model matact-grpo --base-url http://localhost:8000/v1 \
        --temperature 1.0 --out data/self_train/round1
    # adaptation round after the environment update (v2)
    python scripts/self_train.py --policy openai --model matact --tasks data/benchmark/env_shift_train.jsonl \
        --out data/self_train/env_v2
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matact.agents.llm import OpenAIChatPolicy  # noqa: E402
from matact.pipeline.export import LLAMAFACTORY_DATASET_INFO, to_messages, to_sharegpt  # noqa: E402
from matact.pipeline.quality import GAMMA_SELF  # noqa: E402
from matact.pipeline.selftrain import next_training_set, rollout_candidates  # noqa: E402


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="data")
    ap.add_argument("--tasks", default=None, help="task file (default: data/benchmark/train.jsonl)")
    ap.add_argument("--policy", default="openai", choices=["openai"])
    ap.add_argument("--model", default="matact")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=GAMMA_SELF)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-cold", action="store_true",
                    help="write only the filtered rollouts (e.g. teacher trajectories for the budget experiment)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = Path(args.data)
    tasks = read_jsonl(args.tasks or data / "benchmark" / "train.jsonl")
    if args.limit:
        tasks = random.Random(args.seed).sample(tasks, min(args.limit, len(tasks)))

    def make_policy(task, k):
        return OpenAIChatPolicy(args.model, args.base_url, temperature=args.temperature, top_p=args.top_p,
                                seed=args.seed * 1000 + k)

    accepted, stats = rollout_candidates(tasks, make_policy, args.candidates, args.threshold, log_every=25)
    cold = read_jsonl(data / "seed" / "retained_trajectories.jsonl")
    index = {t["task_id"]: t for t in read_jsonl(data / "benchmark" / "train.jsonl") +
             read_jsonl(data / "seed" / "extra_tasks.jsonl")}
    index.update({t["task_id"]: t for t in tasks})
    combined = accepted if args.no_cold else next_training_set(cold, accepted)
    messages = [to_messages(index[t["task_id"]], t) for t in combined]
    out = Path(args.out)
    write_jsonl(out / "self_generated.jsonl", accepted)
    write_jsonl(out / "train_messages.jsonl", messages)
    write_jsonl(out / "sft_sharegpt.jsonl", (to_sharegpt(m) for m in messages))
    (out / "dataset_info.json").write_text(json.dumps(LLAMAFACTORY_DATASET_INFO, indent=2) + "\n")
    stats.update(tasks=len(tasks), cold_start=len(cold), next_training_set=len(combined))
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
