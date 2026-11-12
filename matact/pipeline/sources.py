"""Seed-pool sources: expert workflows, historical agent logs, failure-repair traces and synthetic executions.

Each source is produced by executing its policy in the environment and writing the trace in that
source's raw log format. Raw logs contain the defects that normalization and quality control handle:
legacy tool names, multi-action messages, missing observations, corrupted calls, interrupted traces,
unsupported claims and duplicates.
"""
from __future__ import annotations

import copy
import json

from ..agents.base import Style
from ..agents.policies import ScriptedPolicy
from ..protocol import format_answer, format_call, format_observation, parse
from ..rollout import run_episode
from .aliases import LEGACY_ARG_NAMES, LEGACY_TOOL_NAMES

SOURCE_SIZES = {"expert": 1500, "agent": 2000, "repair": 1200, "synthetic": 2400}
# Released inventory per source (Supplementary Table 3).
SOURCE_RETAINED = {"expert": 1260, "agent": 1180, "repair": 840, "synthetic": 1620}


def to_raw(record, trace_id, source, task_id, rng, legacy=False, p_merge=0.0):
    """Render an executed episode as a raw log (assistant/tool messages only)."""
    messages, steps, i = [], record["steps"], 0
    while i < len(steps):
        group = [steps[i]]
        if (p_merge and i + 1 < len(steps) and steps[i].get("action") and steps[i + 1].get("action")
                and rng.random() < p_merge):
            group.append(steps[i + 1])
        blocks = []
        for k, step in enumerate(group):
            action = step.get("action")
            if action is None:  # policy text that failed to parse is kept verbatim
                blocks.append(step["assistant"])
                continue
            name, args = action["name"], action["arguments"]
            if legacy:
                name = LEGACY_TOOL_NAMES.get(name, name)
                args = {LEGACY_ARG_NAMES.get(a, a): v for a, v in args.items()}
            blocks.append(format_call(name, args, step.get("rationale", "") if k == 0 else ""))
        messages.append({"role": "assistant", "content": "\n".join(blocks)})
        messages += [{"role": "tool", "content": format_observation(s["observation"])} for s in group]
        i += len(group)
    if record.get("answer") is not None:
        final = parse(record["messages"][-1]["content"])
        messages.append({"role": "assistant", "content": format_answer(record["answer"], final.rationale)})
    return {"trace_id": trace_id, "source": source, "task_id": task_id, "messages": messages,
            "meta": {"policy": record["policy"], "status": record["status"], "legacy_names": legacy}}


def inject_defect(raw, kind, rng):
    """Apply one recording defect to a raw log in place; returns the defect name."""
    msgs = raw["messages"]
    tool_idx = [i for i, m in enumerate(msgs) if m["role"] == "tool"]
    if kind == "interrupted" and len(msgs) > 2:
        del msgs[-2:]
    elif kind == "missing_observation" and tool_idx:
        del msgs[rng.choice(tool_idx)]
    elif kind == "corrupt_call":
        calls = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and "<tool_call>" in m["content"]]
        if calls:
            i = rng.choice(calls)
            msgs[i]["content"] = msgs[i]["content"].replace('"arguments": {', '"arguments": {{', 1)
    elif kind in ("unsupported_claim", "unverifiable_artifact") and msgs and "<answer>" in msgs[-1]["content"]:
        text = msgs[-1]["content"]
        if kind == "unsupported_claim":
            claim = rng.choice([" This agrees with the experimental value of {:.3f}.", " The estimated uncertainty is {:.3f}.",
                                " A literature value of {:.2f} was also reported."])
            text = text.replace("\n</answer>", claim.format(rng.uniform(0.5, 9.0)) + "\n</answer>")
        else:
            text = text.replace("\n</answer>", " Full log: workspace://logs/session_full.txt.\n</answer>")
        msgs[-1]["content"] = text
    raw["meta"]["defect"] = kind
    return kind


def _style(rng, style_key="expert"):
    if style_key == "expert":
        return Style(cutoff=rng.choice([520.0, 520.0, 560.0, 600.0]), cpu_factor=rng.choice([1.2, 1.3, 1.5]),
                     epochs=rng.choice([100, 100, 150]), seed=rng.randrange(10 ** 6))
    return Style(cutoff=rng.choice([520.0] * 6 + [450.0, 400.0]), k_scale=rng.choice([1.0] * 6 + [0.8, 0.6]),
                 smearing_metals=rng.random() > 0.15, split_dataset=rng.random() > 0.10,
                 verify_writes=rng.random() > 0.15, unit_slip=rng.random() < 0.10,
                 cpu_factor=rng.choice([1.2, 1.5, 2.0]), epochs=rng.choice([50, 100, 100, 200]),
                 seed=rng.randrange(10 ** 6))


def _draw(tasks, n, rng):
    """Sample n tasks, without replacement while possible."""
    order = list(tasks)
    rng.shuffle(order)
    return [order[k % len(order)] for k in range(n)]


def expert_traces(tasks, n, rng):
    out = []
    for k, task in enumerate(_draw(tasks, n, rng)):
        record, _ = run_episode(task, ScriptedPolicy(_style(rng)))
        raw = to_raw(record, f"expert-{k:05d}", "expert", task["task_id"], rng)
        r = rng.random()
        if r < 0.05:
            inject_defect(raw, "interrupted", rng)
        elif r < 0.09:
            inject_defect(raw, "missing_observation", rng)
        elif r < 0.12:
            inject_defect(raw, "unsupported_claim", rng)
        elif r < 0.16 and out:
            raw = copy.deepcopy(rng.choice(out[-50:]))
            raw.update(trace_id=f"expert-{k:05d}")
            raw["meta"]["defect"] = "duplicate"
        out.append(raw)
    return out


def agent_traces(tasks, n, rng):
    """Historical agent logs: recorded runs of the scripted policy on live tasks.

    Logs are written as they were recorded, with whatever defects the sessions contained: legacy
    tool names from older agent stacks, multi-action messages, and a small share of interrupted
    or corrupted recordings. Sessions that produced no final answer are kept as recorded; the
    quality-control stage drops them.
    """
    out = []
    for k, task in enumerate(_draw(tasks, n, rng)):
        record, _ = run_episode(task, ScriptedPolicy(_style(rng, "live")))
        raw = to_raw(record, f"agent-{k:05d}", "agent", task["task_id"], rng, legacy=rng.random() < 0.6, p_merge=0.15)
        r = rng.random()
        if r < 0.035:
            inject_defect(raw, "unsupported_claim", rng)
        elif r < 0.060:
            inject_defect(raw, "corrupt_call", rng)
        elif r < 0.095:
            inject_defect(raw, "interrupted", rng)
        elif r < 0.115:
            inject_defect(raw, "unverifiable_artifact", rng)
        out.append(raw)
    return out


def with_fault(task, rng):
    """Copy of a training task with an injected tool failure (used to enlarge the repair source)."""
    t = copy.deepcopy(task)
    fault = {"first_principles": [("dft.run", "NON_CONVERGENCE"), ("dft.run", "TIMEOUT")],
             "ml_potential": [("mlp.train", "UNSTABLE_TRAINING")],
             "database": [("database.update", "SCHEMA_CONFLICT")]}.get(t["category"])
    if not fault:
        return None
    tool, code = rng.choice(fault)
    t["faults"] = [{"tool": tool, "code": code}]
    t["task_id"] = f"{task['task_id']}-F{code[:3]}"
    t["difficulty"] = "tool_failure"
    if code == "TIMEOUT":
        t["subtype"] = "dft_timeout"
    return t


def repair_traces(tasks, n, rng):
    """Failure-repair traces on tool-failure tasks (plus fault-injected copies of regular tasks)."""
    pool = [t for t in tasks if t["difficulty"] == "tool_failure"]
    regular = [t for t in tasks if t["difficulty"] == "regular" and t["subtype"] in
               ("dft_property", "mlp_finetune", "db_insert")]
    injected = [x for x in (with_fault(t, rng) for t in regular) if x is not None]
    out = []
    for k, task in enumerate(_draw(pool + injected, n, rng)):
        record, _ = run_episode(task, ScriptedPolicy(_style(rng)))
        raw = to_raw(record, f"repair-{k:05d}", "repair", task["task_id"], rng)
        r = rng.random()
        if r < 0.060:
            failed = [i for i, m in enumerate(raw["messages"]) if m["role"] == "tool" and '"status": "error"' in m["content"]]
            if failed:  # give up after the failure and claim a result anyway
                del raw["messages"][failed[0] + 1:]
                raw["messages"].append({"role": "assistant", "content": format_answer(
                    f"The calculation finished successfully; the requested value is {rng.uniform(0.5, 6):.4f}.")})
                raw["meta"]["defect"] = "unrepaired_claim"
        elif r < 0.125:
            inject_defect(raw, "interrupted", rng)
        out.append(raw)
    return out, injected


def synthetic_traces(tasks, n, rng):
    out = []
    for k, task in enumerate(tasks[:n]):
        record, _ = run_episode(task, ScriptedPolicy(_style(rng)))
        raw = to_raw(record, f"synthetic-{k:05d}", "synthetic", task["task_id"], rng)
        if rng.random() < 0.05:
            inject_defect(raw, "interrupted", rng)
        out.append(raw)
    return out


def dumps_raw(raw) -> str:
    return json.dumps(raw, ensure_ascii=False)
