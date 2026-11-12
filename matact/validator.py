"""Optional runtime action validator (rule-based or LLM-based) used by the router in ``rollout``.

The validator checks a proposed action against the current tool registry and
environment state before execution. It never executes actions and never reads
reference answers. Decisions: ACCEPT, REPAIR (with ``repaired_action``) or REJECT,
plus a risk level (LOW, MEDIUM, HIGH).
"""
from __future__ import annotations

import json
import urllib.request

from .env.tools import REGISTRY, check_arguments
from .pipeline.aliases import ARG_ALIASES, TOOL_ALIASES
from .protocol import VALIDATOR_PROMPT

_ENUM_FIXES = {"vasp": "poscar", "CIF": "cif", "extxyz": "xyz", "POSCAR": "poscar", "XYZ": "xyz"}
_NEEDS = {"structure.validate": ("uri", ("structure", "validated_structure")),
          "structure.convert": ("uri", ("structure", "validated_structure")),
          "dft.prepare": ("structure_uri", ("validated_structure",)), "dft.run": ("input_uri", ("dft_input",)),
          "result.extract": ("output_uri", ("dft_output",)), "mlp.evaluate": ("model_uri", ("mlp_model",))}


def decision(kind, reason, risk="LOW", repaired=None):
    out = {"decision": kind, "reason": reason, "risk": risk}
    if repaired is not None:
        out["repaired_action"] = repaired
    return out


class RuleValidator:
    """Deterministic validator implementing the checks of the validator prompt in order."""
    name = "rule"

    def check_answer(self, answer: str, steps, task) -> dict:
        """Reject final answers whose numbers cannot be traced to the query, actions or observations."""
        from .verify.answer import provenance
        _, missing = provenance(task, steps, answer)
        if missing:
            return decision("REJECT", f"unsupported numerical claims {missing[:5]}; report only observed values")
        return decision("ACCEPT", "all numerical claims are grounded")

    def check(self, action: dict, env, history) -> dict:
        name = TOOL_ALIASES.get(action["name"], action["name"])
        args = {ARG_ALIASES.get(k, k): v for k, v in action["arguments"].items()}
        repaired = name != action["name"] or args != action["arguments"]
        if name not in REGISTRY or name not in env.available_tools:
            return decision("REJECT", f"{action['name']} is not an available tool; available: {env.available_tools}")
        if name in env.deprecated:
            return decision("REJECT", f"{name} is deprecated; use database.query with fields=['structure_uri']")
        for key, value in list(args.items()):
            if isinstance(value, str) and value in _ENUM_FIXES and key == "format":
                args[key], repaired = _ENUM_FIXES[value], True
        if name == "dft.prepare" and isinstance(args.get("k_mesh"), str):
            try:
                args["k_mesh"], repaired = [int(x) for x in args["k_mesh"].lower().split("x")], True
            except ValueError:
                pass
        message = check_arguments(name, args)
        if message:
            return decision("REJECT", message)
        if name in _NEEDS:
            key, kinds = _NEEDS[name]
            art, err = env.get(args[key], kinds)
            if err:
                return decision("REJECT", err["message"])
            if name == "dft.prepare":
                from .env import materials
                if materials.by_id(art["material_id"]).metallic and args.get("smearing", "none") == "none":
                    args["smearing"], repaired = "mv", True
            if name == "dft.run":
                remaining = env.resources.get("cpu_hours", 0.0)
                need = art["estimate"]
                if need > remaining:
                    return decision("REJECT", f"the run needs about {need} CPU hours but only {remaining:.2f} remain",
                                    risk="HIGH")
                if args["max_cpu_hours"] > remaining:
                    args["max_cpu_hours"], repaired = round(remaining, 2), True
                elif args["max_cpu_hours"] < need:
                    args["max_cpu_hours"], repaired = round(min(remaining, need * 1.2), 2), True
        for step in history:
            if step.get("action") == {"name": name, "arguments": args} and step["observation"].get("status") == "success":
                return decision("REJECT", f"redundant repetition of {step['observation']['observation_id']}")
        if repaired:
            return decision("REPAIR", "normalized names, enumerations or limits", "MEDIUM",
                            {"name": name, "arguments": args})
        return decision("ACCEPT", "action is consistent with the environment state")


class LLMValidator:
    """Validator run by a prompted model behind an OpenAI-compatible endpoint (greedy decoding)."""
    name = "llm"

    def __init__(self, model, base_url="http://localhost:8000/v1", api_key="", timeout=120):
        self.model, self.base_url, self.api_key, self.timeout = model, base_url.rstrip("/"), api_key, timeout

    def check(self, action, env, history):
        prompt = (f"{VALIDATOR_PROMPT}\n\n<environment_state>\n{json.dumps(env.public_state())}\n</environment_state>\n"
                  f"Proposed action:\n{json.dumps(action)}")
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0}
        headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})}
        req = urllib.request.Request(self.base_url + "/chat/completions", json.dumps(body).encode(), headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            text = json.load(resp)["choices"][0]["message"]["content"]
        try:
            out = json.loads(text[text.find("{"): text.rfind("}") + 1])
        except ValueError:
            return decision("ACCEPT", "validator output was not valid JSON; action passed through")
        if out.get("decision") not in ("ACCEPT", "REPAIR", "REJECT"):
            return decision("ACCEPT", "validator returned no decision")
        return out
