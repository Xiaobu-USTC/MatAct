#!/usr/bin/env python3
"""Evaluate a policy on MatAct-Bench and report SR, FR, FCR, ESR, ATC and CPS.

    # scripted expert policy (no model needed; checks the harness and the environment)
    python scripts/evaluate.py --suite test --policy expert
    # a trained checkpoint served by vLLM (OpenAI-compatible API)
    python scripts/evaluate.py --suite test ood_chem ood_comp tool_failure --policy openai \
        --model matact --base-url http://localhost:8000/v1 --out results/matact.json
    # with the runtime validator and router
    python scripts/evaluate.py --suite stress --policy openai --model matact --validator rule
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matact.agents.llm import OpenAIChatPolicy  # noqa: E402
from matact.agents.policies import ScriptedPolicy  # noqa: E402
from matact.eval.metrics import evaluate_policy, report  # noqa: E402
from matact.validator import LLMValidator, RuleValidator  # noqa: E402

SPLITS = ("train", "valid", "test")
SUITES = ("ood_chem", "ood_comp", "paraphrase", "tool_failure", "env_shift", "stress")


def load_suite(data: Path, name: str):
    path = data / "benchmark" / (f"{name}.jsonl" if name in SPLITS else f"suites/{name}.jsonl")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def make_factory(args):
    if args.policy == "expert":
        return lambda task: ScriptedPolicy()
    kwargs = json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else None
    return lambda task: OpenAIChatPolicy(args.model, args.base_url, temperature=args.temperature, top_p=args.top_p,
                                         max_tokens=args.max_tokens, use_tools_param=not args.no_tools_param,
                                         sanitize_names=args.sanitize_tool_names, chat_template_kwargs=kwargs)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="data")
    ap.add_argument("--suite", nargs="+", default=["test"], choices=SPLITS + SUITES)
    ap.add_argument("--policy", default="openai", choices=["expert", "openai"])
    ap.add_argument("--model", default="matact")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--no-tools-param", action="store_true", help="do not send the tools list in the request")
    ap.add_argument("--sanitize-tool-names", action="store_true", help="send structure.search as structure__search")
    ap.add_argument("--chat-template-kwargs", default=None,
                    help='JSON passed to vLLM, e.g. \'{"enable_thinking": false}\' for an untrained Qwen3 model')
    ap.add_argument("--validator", default="none", choices=["none", "rule", "llm"])
    ap.add_argument("--validator-model", default=None)
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N tasks of each suite")
    ap.add_argument("--reference", default=None, help="results JSON of a reference agent used to normalize CPS")
    ap.add_argument("--out", default=None, help="write metrics (JSON) and trajectories (JSONL) here")
    args = ap.parse_args()

    validator = {"none": None, "rule": RuleValidator()}.get(args.validator)
    if args.validator == "llm":
        validator = LLMValidator(args.validator_model or args.model, args.base_url)
    reference = json.loads(Path(args.reference).read_text()) if args.reference else {}
    factory = make_factory(args)
    results, trajectories = {}, []
    for name in args.suite:
        tasks = load_suite(Path(args.data), name)
        tasks = tasks[:args.limit] if args.limit else tasks
        print(f"[{name}] {len(tasks)} tasks with policy={args.policy} validator={args.validator}")
        rows = evaluate_policy(tasks, factory, validator, log_every=50)
        ref_cps = reference.get("suites", {}).get(name, {}).get("overall", {}).get("CPS_units")
        results[name] = report(rows, ref_cps)
        print(json.dumps(results[name]["overall"]))
        trajectories += [dict(r, suite=name) for r in rows]
    output = {"config": vars(args), "suites": results}
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        with out.with_suffix(".trajectories.jsonl").open("w", encoding="utf-8") as fh:
            for row in trajectories:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {out} and {out.with_suffix('.trajectories.jsonl')}")


if __name__ == "__main__":
    main()
