"""verl integration: verifiable reward computed by replaying the policy's actions in the environment.

Use with ``custom_reward_function.path=matact/integrations/verl_reward.py`` and
``custom_reward_function.name=compute_score``. ``ground_truth`` is the full task JSON
written by ``scripts/build_dataset.py`` (data/rl/*.jsonl).
"""
from __future__ import annotations

import json
import re

from ..env.common import MAX_ACTIONS
from ..env.environment import ScientificEnvironment
from ..verify.scoring import score

_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>|<answer>\s*(.*?)\s*</answer>", re.S)
_STRIP = re.compile(r"<tool_response>.*?</tool_response>|<think>.*?</think>", re.S)


def record_from_text(task: dict, text: str):
    """Rebuild a trajectory from a decoded multi-turn response and re-execute its actions."""
    env = ScientificEnvironment(task)
    steps, answer, executed = [], None, 0
    for call, final in _BLOCK.findall(_STRIP.sub("", text or "")):
        if len(steps) >= MAX_ACTIONS:
            break
        if not call:
            answer = final.strip()
            break
        try:
            obj = json.loads(call)
            args = obj.get("arguments", {})
            args = json.loads(args) if isinstance(args, str) else args
            if not isinstance(obj.get("name"), str) or not isinstance(args, dict):
                raise ValueError("bad call")
        except (ValueError, AttributeError):
            steps.append({"action": None, "parse_error": "invalid tool call", "assistant": call,
                          "observation": {"observation_id": f"obs-parse-{len(steps) + 1}", "status": "error",
                                          "code": "PARSE", "message": "invalid tool call"}})
            continue
        executed += 1
        action = {"name": obj["name"], "arguments": args}
        steps.append({"action": action, "observation": env.step(action, f"obs-{executed}")})
    status = "answer" if answer is not None else "max_actions"
    return {"task_id": task["task_id"], "status": status, "steps": steps, "answer": answer}, env


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """MatAct verifiable, cost-aware reward R(tau) (Supplementary Eq. S3)."""
    task = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    record, env = record_from_text(task, solution_str)
    sc = score(task, record, env)
    return {"score": sc["reward"], "success": float(sc["success"]), "utility": sc["utility"],
            "factual": float(sc["factual"]), "n_calls": sc["n_calls"], "cost": sc["cost"]}


def compute_score_generic(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Generic tool-RL baseline: formatting, execution-success and final-answer rewards only
    (no scientific-validity, grounding or cost terms, Methods)."""
    task = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    record, env = record_from_text(task, solution_str)
    sc = score(task, record, env)
    formatting = sc["n_format_ok"] / max(1, sc["n_turns"])
    execution = sc["n_exec_ok"] / sc["n_calls"] if sc["n_calls"] else 0.0
    reward = (formatting + execution + float(sc["answer_correct"])) / 3.0
    return {"score": reward, "success": float(sc["success"]), "factual": float(sc["factual"]),
            "n_calls": sc["n_calls"], "cost": sc["cost"]}
