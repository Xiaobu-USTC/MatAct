"""Episode runner: policy <-> environment loop with the optional validator and router."""
from __future__ import annotations

from .env.common import MAX_ACTIONS
from .env.environment import ScientificEnvironment
from .protocol import POLICY_SYSTEM_PROMPT, format_observation, parse, user_message

MAX_REJECTIONS = 3
MAX_PARSE_FAILURES = 3


def initial_messages(task: dict, state: dict) -> list:
    return [{"role": "system", "content": POLICY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message(task["query"], state)}]


def run_episode(task: dict, policy, validator=None, max_actions: int = MAX_ACTIONS) -> dict:
    """Roll out ``policy`` on ``task``. Returns the trajectory record and the final environment."""
    env = ScientificEnvironment(task)
    state = env.public_state()
    messages = initial_messages(task, state)
    policy.start(task, state)
    steps, observation, answer, status = [], None, None, "max_actions"
    parse_failures = rejections = 0
    confirm = None  # high-risk action awaiting re-issue
    while len(steps) < max_actions:
        text = policy.act(messages, observation)
        messages.append({"role": "assistant", "content": text})
        parsed = parse(text)
        oid = f"obs-{len(steps) + 1}"
        step = {"assistant": text, "rationale": parsed.rationale, "action": None}
        if parsed.kind == "answer":
            if validator is not None:
                review = validator.check_answer(parsed.answer, steps, task) if hasattr(validator, "check_answer") else None
                if review and review["decision"] == "REJECT" and rejections < MAX_REJECTIONS - 1:
                    rejections += 1
                    observation = {"observation_id": oid, "status": "rejected", "code": "VALIDATOR_REJECT",
                                   "message": review["reason"]}
                    step.update(observation=observation, validator=review)
                    steps.append(step)
                    messages.append({"role": "tool", "content": format_observation(observation)})
                    continue
            answer, status = parsed.answer, "answer"
            break
        if parsed.kind == "error":
            parse_failures += 1
            observation = {"observation_id": oid, "status": "error", "code": "PARSE", "message": parsed.error}
            step.update(observation=observation, parse_error=parsed.error)
        else:
            parse_failures = 0
            action = parsed.action
            review = None
            if validator is not None and action != confirm:
                review = validator.check(action, env, steps)
            confirm = None
            if review is None or review["decision"] == "ACCEPT":
                rejections = 0
                observation = env.step(action, oid)
                step.update(action=action, observation=observation)
            elif review["decision"] == "REPAIR":
                rejections = 0
                executed = review["repaired_action"]
                observation = dict(env.step(executed, oid), validator={"decision": "REPAIR", "reason": review["reason"],
                                                                       "executed_action": executed})
                step.update(action=executed, proposed_action=action, observation=observation)
            else:
                rejections += 1
                code = "VALIDATOR_HIGH_RISK" if review.get("risk") == "HIGH" else "VALIDATOR_REJECT"
                if code == "VALIDATOR_HIGH_RISK":
                    confirm = action
                observation = {"observation_id": oid, "status": "rejected", "code": code, "message": review["reason"]}
                step.update(proposed_action=action, observation=observation)
            if review is not None:
                step["validator"] = review
        steps.append(step)
        messages.append({"role": "tool", "content": format_observation(observation)})
        if parse_failures >= MAX_PARSE_FAILURES:
            status = "parse_failures"
            break
        if rejections >= MAX_REJECTIONS:
            status = "validator_rejections"
            break
    return {"task_id": task["task_id"], "policy": getattr(policy, "name", "policy"), "status": status,
            "steps": steps, "answer": answer, "messages": messages}, env


def replay(task: dict, steps: list):
    """Re-execute the logged actions in a fresh environment; return (env, all observations matched)."""
    env = ScientificEnvironment(task)
    matched = True
    for i, step in enumerate(steps, start=1):
        action = step.get("action")
        if action is None:
            continue
        obs = env.step(action, f"obs-{i}")
        logged = {k: v for k, v in step["observation"].items() if k != "validator"}
        if obs != logged:
            matched = False
    return env, matched
