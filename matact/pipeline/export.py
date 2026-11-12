"""Exporters: canonical chat messages, LLaMA-Factory ShareGPT, and verl RL prompts.

The tool list given to the model (the chat-template ``tools``) is always the full 14-tool
registry; which tools a task actually provides is stated in ``<environment_state>``
(policy rule 1). Keeping the list fixed also gives every verl row the same ``tools_kwargs``
keys, which Parquet requires.
"""
from __future__ import annotations

import json

from ..env.environment import ScientificEnvironment
from ..env.tools import TOOLS, openai_schema
from ..protocol import POLICY_SYSTEM_PROMPT, format_answer, format_call, format_observation, user_message

TOOL_NAMES = [t.name for t in TOOLS]


def initial_user_message(task: dict) -> str:
    return user_message(task["query"], ScientificEnvironment(task).public_state())


def to_messages(task: dict, traj: dict) -> dict:
    """Canonical SFT record: OpenAI-style messages plus the tool schemas (rendered by the chat template)."""
    messages = [{"role": "system", "content": POLICY_SYSTEM_PROMPT},
                {"role": "user", "content": initial_user_message(task)}]
    for step in traj["steps"]:
        action = step["action"]
        messages.append({"role": "assistant", "content": format_call(action["name"], action["arguments"], step.get("rationale", ""))})
        messages.append({"role": "tool", "content": format_observation(step["observation"])})
    messages.append({"role": "assistant", "content": format_answer(traj["answer"], traj.get("answer_rationale", ""))})
    return {"id": traj["trajectory_id"], "task_id": task["task_id"], "source": traj["source"],
            "category": task["category"], "difficulty": task["difficulty"], "utility": traj["scores"]["utility"],
            "messages": messages, "tools": openai_schema()}


def to_sharegpt(record: dict) -> dict:
    """LLaMA-Factory 'sharegpt' format with function_call / observation turns."""
    conv = []
    for m in record["messages"][1:]:
        if m["role"] == "user":
            conv.append({"from": "human", "value": m["content"]})
        elif m["role"] == "tool":
            conv.append({"from": "observation", "value": m["content"]})
        elif "<tool_call>" in m["content"]:
            body = m["content"].split("<tool_call>", 1)[1].split("</tool_call>", 1)[0].strip()
            conv.append({"from": "function_call", "value": body})
        else:
            conv.append({"from": "gpt", "value": m["content"]})
    return {"conversations": conv, "system": record["messages"][0]["content"],
            "tools": json.dumps(record["tools"], ensure_ascii=False)}


LLAMAFACTORY_DATASET_INFO = {
    "matact_sft": {"file_name": "sft_sharegpt.jsonl", "formatting": "sharegpt",
                   "columns": {"messages": "conversations", "system": "system", "tools": "tools"}},
}


def env_spec(task: dict) -> dict:
    """What the environment needs to reproduce a task (includes hidden faults, excludes references)."""
    return {k: task[k] for k in ("task_id", "env_version", "initial_state", "available_tools", "faults")}


def to_rl_row(task: dict, index: int) -> dict:
    """verl-style RL prompt. ``ground_truth`` carries the full task so that the reward can replay actions.

    Every row lists all registry tools in ``tools_kwargs``; the environment specification is passed
    once, to the first tool, and the tools of one rollout share that environment.
    """
    spec = json.dumps(env_spec(task), ensure_ascii=False, sort_keys=True)
    tools_kwargs = {name: {"create_kwargs": {"task_id": task["task_id"], "env_spec": spec if i == 0 else ""}}
                    for i, name in enumerate(TOOL_NAMES)}
    return {
        "data_source": "matact",
        "prompt": [{"role": "system", "content": POLICY_SYSTEM_PROMPT},
                   {"role": "user", "content": initial_user_message(task)}],
        "ability": "scientific_tool_use",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(task, ensure_ascii=False, sort_keys=True)},
        "extra_info": {"index": index, "task_id": task["task_id"], "split": task["split"],
                       "category": task["category"], "difficulty": task["difficulty"],
                       "need_tools_kwargs": True, "tools_kwargs": tools_kwargs},
    }


def verl_tool_config() -> dict:
    """Tool configuration for verl multi-turn rollouts (JSON is valid YAML)."""
    return {"tools": [{"class_name": "matact.integrations.verl_tools.MatActTool", "config": {"type": "native"},
                       "tool_schema": schema} for schema in openai_schema()]}
