"""Action protocol: prompt templates, message formats and parsing of policy outputs.

A policy turn is either one structured tool call or one grounded final answer:

    <tool_call>
    {"name": "structure.validate", "arguments": {"uri": "workspace://..."}}
    </tool_call>

    <answer>
    ... grounded final answer citing observation IDs or artifact URIs ...
    </answer>

Optional short rationale text may precede either block.
"""
from __future__ import annotations

import json
import re
from collections import namedtuple

POLICY_SYSTEM_PROMPT = """You are MatAct, a materials-computing agent operating in a typed scientific
software environment.

Your objective is to complete the user's request with the smallest reliable
sequence of tool calls. Before calling a tool, check its schema, required
arguments, artifact dependencies, and current resource budget.

Rules:
1. Use only tools listed in <environment_state>.
2. Emit at most one structured tool call at a time.
3. Never invent an artifact, value, model, record, or execution status.
4. If a call fails, inspect the observation and issue a corrected call or
   return a bounded, grounded failure report.
5. Preserve units and precision from observations.
6. Every final claim must be grounded in an observation or artifact URI.
7. Stop when the success specification is satisfied; avoid repeated queries.

Output exactly one of:
<tool_call>{"name": ..., "arguments": {...}}</tool_call>
<answer>grounded final answer with source artifacts</answer>"""

VALIDATOR_PROMPT = """Validate the proposed action against the current environment.
Return JSON with fields: decision (ACCEPT, REPAIR, or REJECT), reason,
repaired_action (only for REPAIR), and risk (LOW, MEDIUM, or HIGH).

Check in order: exactly one call; available tool; required arguments and types;
artifact compatibility; resource budget; redundant repetition; and unsupported
claims. Do not execute the action and do not infer hidden reference answers."""

REPAIR_PROMPT = """The previous scientific tool call failed.
User request: {query}
Previous action: {previous_action}
Observation: {observation}
Available tools: {tool_schemas}
Produce one corrected action. Do not repeat the same invalid call. If the task
cannot be completed, return a grounded answer identifying the missing resource
or failed precondition."""

JUDGE_PROMPT = """Judge the trajectory using only the query, observations, final answer, and
success specification. Return JSON:
{"task_complete": true/false, "scientifically_supported": true/false,
 "missing_evidence": [...], "unsupported_claims": [...],
 "short_justification": "..."}
Do not infer hidden values. A plausible unsupported answer is incorrect."""

_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S)
_THINK = re.compile(r"<think>.*?</think>", re.S)

Parsed = namedtuple("Parsed", "kind action answer rationale error")


def render_state(state: dict) -> str:
    return "<environment_state>\n" + json.dumps(state, ensure_ascii=False) + "\n</environment_state>"


def user_message(query: str, state: dict) -> str:
    return f"{query}\n\n{render_state(state)}"


def format_call(name: str, arguments: dict, rationale: str = "") -> str:
    block = "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False) + "\n</tool_call>"
    return f"{rationale.strip()}\n{block}" if rationale else block


def format_answer(text: str, rationale: str = "") -> str:
    block = f"<answer>\n{text.strip()}\n</answer>"
    return f"{rationale.strip()}\n{block}" if rationale else block


def format_observation(obs: dict) -> str:
    return json.dumps(obs, ensure_ascii=False)


def parse(text: str) -> Parsed:
    text = _THINK.sub("", text or "").strip()
    calls, answers = _CALL.findall(text), _ANSWER.findall(text)
    if calls and answers:
        return Parsed("error", None, None, "", "emit either one tool call or one answer, not both")
    if len(calls) > 1:
        return Parsed("error", None, None, "", "exactly one tool call is allowed per turn")
    if calls:
        rationale = text[:text.find("<tool_call>")].strip()
        try:
            obj = json.loads(calls[0])
        except json.JSONDecodeError as exc:
            return Parsed("error", None, None, rationale, f"tool call is not valid JSON: {exc.msg}")
        args = obj.get("arguments", {}) if isinstance(obj, dict) else None
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = None
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or not isinstance(args, dict):
            return Parsed("error", None, None, rationale, "tool call needs a string 'name' and an object 'arguments'")
        return Parsed("call", {"name": obj["name"], "arguments": args}, None, rationale, "")
    if answers:
        return Parsed("answer", None, answers[0].strip(), text[:text.find("<answer>")].strip(), "")
    if "<tool_call>" in text or "<answer>" in text:
        return Parsed("error", None, None, "", "unbalanced <tool_call> or <answer> tags")
    return Parsed("error", None, None, "", "no <tool_call> or <answer> block found")
