"""Normalization of raw traces into canonical trajectories (Supplementary Note 4).

Ordered rules: (1) map tool names and argument keys to the canonical registry;
(2) keep observations, status codes, units and artifact URIs exactly; (3) split
multi-action messages when the number of calls equals the number of following
observations; (4) observations are kept as separate tool turns so that they can
be masked during supervised training; (5) attach failure-repair edges; (6) drop
traces with missing final answers, unrecoverable parser errors or missing observations.
"""
from __future__ import annotations

import json
import re
from collections import Counter

from ..env.tools import REGISTRY
from .aliases import ARG_ALIASES, TOOL_ALIASES

_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)


class Rejected(Exception):
    pass


def _canonical(call: dict, stats: Counter) -> dict:
    name = TOOL_ALIASES.get(call["name"], call["name"])
    stats["renamed_tools"] += name != call["name"]
    args = {}
    for key, value in call.get("arguments", {}).items():
        new = ARG_ALIASES.get(key, key)
        stats["renamed_arguments"] += new != key
        args[new] = value
    return {"name": name, "arguments": args}


def repair_edges(steps):
    edges = []
    for i, s in enumerate(steps):
        obs = s.get("observation", {})
        if obs.get("status") != "error" or obs.get("code") in ("NO_MATCH", "EMPTY_RESULT"):
            continue
        name = (s.get("action") or {}).get("name")
        family = REGISTRY[name].family if name in REGISTRY else None
        for j in range(i + 1, len(steps)):
            later = steps[j]
            lname = (later.get("action") or {}).get("name")
            if later["observation"].get("status") == "success" and lname in REGISTRY and \
                    (lname == name or REGISTRY[lname].family == family):
                edges.append({"failed_step": i, "repair_step": j, "code": obs.get("code")})
                break
    return edges


def normalize(raw: dict) -> dict:
    """Return a canonical trajectory or raise ``Rejected`` with the reason."""
    stats, steps = Counter(), []
    msgs, i, answer, answer_rationale = raw["messages"], 0, None, ""
    while i < len(msgs):
        msg = msgs[i]
        if msg["role"] != "assistant":
            raise Rejected("observation without a preceding call")
        if answer is not None:
            raise Rejected("turns after the final answer")
        content = msg["content"]
        blocks, answers = _CALL.findall(content), _ANSWER.findall(content)
        j = i + 1
        while j < len(msgs) and msgs[j]["role"] == "tool":
            j += 1
        observations = [json.loads(m["content"]) for m in msgs[i + 1:j]]
        if answers and not blocks:
            if observations:
                raise Rejected("observation after the final answer")
            answer = answers[0].strip()
            answer_rationale = content[:content.find("<answer>")].strip()
            i = j
            continue
        if not blocks:
            if len(observations) == 1 and observations[0].get("code") == "PARSE":
                steps.append({"rationale": "", "action": None, "parse_error": observations[0]["message"],
                              "assistant": content, "observation": observations[0]})
                i = j
                continue
            raise Rejected("turn without a call or an answer")
        if len(observations) != len(blocks):
            raise Rejected("missing observation" if len(observations) < len(blocks) else "ambiguous action boundaries")
        stats["split_messages"] += len(blocks) > 1
        rationale = content[:content.find("<tool_call>")].strip()
        for k, (block, obs) in enumerate(zip(blocks, observations)):
            try:
                call = json.loads(block)
            except json.JSONDecodeError:
                raise Rejected("unrecoverable parser error")
            if not isinstance(call, dict) or not isinstance(call.get("name"), str):
                raise Rejected("unrecoverable parser error")
            steps.append({"rationale": rationale if k == 0 else "", "action": _canonical(call, stats), "observation": obs})
        i = j
    if answer is None:
        raise Rejected("missing final answer")
    if not answer.strip():
        raise Rejected("empty final answer")
    return {"trajectory_id": raw["trace_id"], "task_id": raw["task_id"], "source": raw["source"], "steps": steps,
            "answer": answer, "answer_rationale": answer_rationale, "status": "answer",
            "repair_edges": repair_edges(steps), "normalization": dict(stats)}
