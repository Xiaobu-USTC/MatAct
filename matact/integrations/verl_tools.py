"""verl integration: the MatAct environment exposed as verl multi-turn tools.

All 14 tools share one environment per rollout (keyed by verl's instance/request id).
Every row of data/rl/*.jsonl lists all tools in ``extra_info.tools_kwargs``; the environment
specification (initial state and hidden faults) is passed only to the first tool, as
``create_kwargs.env_spec``. Generate the tool configuration with ``scripts/build_dataset.py``
(data/rl/verl_tools.yaml).

Written against verl's ``BaseTool`` interface (create / execute / calc_reward / release);
check the signatures against the verl version you install.
"""
from __future__ import annotations

import json
import uuid

from ..env.environment import ScientificEnvironment

try:  # pragma: no cover - optional dependency
    from verl.tools.base_tool import BaseTool
except ImportError:  # keeps the module importable without verl
    class BaseTool:  # type: ignore[no-redef]
        def __init__(self, config, tool_schema):
            self.config, self.tool_schema = config, tool_schema

try:  # pragma: no cover - newer verl versions return ToolResponse objects
    from verl.tools.schemas import ToolResponse
except ImportError:
    ToolResponse = None

_ENVIRONMENTS = {}


def _tool_name(tool_schema) -> str:
    function = getattr(tool_schema, "function", None)
    if function is not None:
        return function.name
    return tool_schema["function"]["name"]


class MatActTool(BaseTool):
    def __init__(self, config, tool_schema):
        super().__init__(config, tool_schema)
        self.tool_name = _tool_name(tool_schema)

    async def create(self, instance_id=None, env_spec=None, **kwargs):
        instance_id = instance_id or str(uuid.uuid4())
        if env_spec:
            spec = json.loads(env_spec) if isinstance(env_spec, str) else env_spec
            _ENVIRONMENTS[instance_id] = {"env": ScientificEnvironment(spec), "calls": 0}
        if ToolResponse is not None:
            return instance_id, ToolResponse()
        return instance_id

    async def execute(self, instance_id, parameters, **kwargs):
        slot = _ENVIRONMENTS.get(instance_id)
        if slot is None:
            obs = {"status": "error", "code": "NO_ENVIRONMENT", "message": "environment was not created for this rollout"}
        else:
            slot["calls"] += 1
            if isinstance(parameters, str):
                try:
                    parameters = json.loads(parameters)
                except json.JSONDecodeError:
                    parameters = {"__unparsed__": parameters}
            obs = slot["env"].step({"name": self.tool_name, "arguments": parameters}, f"obs-{slot['calls']}")
        text = json.dumps(obs, ensure_ascii=False)
        return (ToolResponse(text=text) if ToolResponse is not None else text), 0.0, {}

    async def calc_reward(self, instance_id, **kwargs):
        return 0.0  # the trajectory-level reward is computed by verl_reward.compute_score

    async def release(self, instance_id, **kwargs):
        _ENVIRONMENTS.pop(instance_id, None)
