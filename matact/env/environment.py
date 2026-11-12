"""The MatAct scientific environment: workspace, registries, resources and dispatch.

``ScientificEnvironment(task).step(action)`` executes one structured action
``{"name": ..., "arguments": {...}}`` and returns an observation dictionary. The
environment is deterministic: replaying the same actions on the same task gives
identical observations, which the verifiers use to re-check logged trajectories.
"""
from __future__ import annotations

import copy
import hashlib
import json

from ..chem import fileio
from . import materials
from .common import ERROR_LABELS, MAX_ACTIONS, error, in_workspace  # noqa: F401  (re-exported)
from .handlers_compute import ComputeHandlers
from .handlers_data import DataHandlers
from .handlers_structure import StructureHandlers
from .tools import REGISTRY, check_arguments

MODELS = {
    "qe-7.2": {"kind": "dft", "versions": ("v1", "v2")},
    "chgnet-0.3": {"kind": "mlp", "cost": 2.6, "max_z": 56, "base_mae": 33.0, "cutoff_A": 5.0,
                   "tasks": ("energy_forces", "relaxation"), "versions": ("v1", "v2")},
    "mace-mp-medium": {"kind": "mlp", "cost": 3.2, "max_z": 94, "base_mae": 27.0, "cutoff_A": 6.0,
                       "tasks": ("energy_forces", "relaxation", "md"), "versions": ("v1", "v2")},
    "mace-ft-matact-v2": {"kind": "mlp", "cost": 2.0, "max_z": 94, "base_mae": 8.5, "cutoff_A": 6.0,
                          "tasks": ("energy_forces", "relaxation", "md"), "versions": ("v2",),
                          "exclude": sorted(materials.OOD_ELEMENTS)},
}
PUBLIC_KEYS = ("type", "format", "formula", "natoms", "elements", "n_structures", "label", "split",
               "material_id", "property", "value", "unit", "code", "calc_type", "model", "status")


class ScientificEnvironment(StructureHandlers, ComputeHandlers, DataHandlers):
    def __init__(self, task: dict):
        self.task = task
        self.version = task.get("env_version", "v1")
        state = task["initial_state"]
        self.available_tools = list(task["available_tools"])
        self.deprecated = {"structure.search"} if self.version == "v2" else set()
        visible = state.get("available_models", list(MODELS))
        self.models = {n: MODELS[n] for n in visible if n in MODELS and self.version in MODELS[n]["versions"]}
        self.resources = {k: float(v) for k, v in state["resource_budget"].items()}
        self.faults = copy.deepcopy(task.get("faults", []))
        self.artifacts, self.userdb, self.trace = {}, {}, []
        self.counters = {}
        for uri, spec in state.get("workspace", {}).items():
            self._materialize(uri, spec)

    # ------------------------------------------------------------------ state
    def public_state(self) -> dict:
        return {
            "env_version": self.version,
            "available_tools": self.available_tools,
            "deprecated_tools": sorted(self.deprecated),
            "artifacts": [{"uri": u, "type": a["type"]} for u, a in self.artifacts.items()],
            "models": [{"name": n, "kind": m["kind"], **({"cost": m["cost"]} if "cost" in m else {})}
                       for n, m in self.models.items()],
            "databases": {"matact_refdb": "reference records (read-only)", "matact_userdb": "user records"},
            "resources": {f"{k}_remaining": round(v, 2) for k, v in self.resources.items()},
            "constraints": {"max_actions": MAX_ACTIONS, "require_provenance": True},
        }

    # --------------------------------------------------------------- dispatch
    def step(self, action, observation_id: str = None) -> dict:
        obs = self._dispatch(action)
        obs = {"observation_id": observation_id or f"obs-{len(self.trace) + 1}", **obs}
        self.trace.append({"action": action, "observation": obs})
        return obs

    def _dispatch(self, action) -> dict:
        if not isinstance(action, dict) or not isinstance(action.get("name"), str):
            return error("INVALID_ARGUMENTS", "action must be an object with a tool name")
        name, args = action["name"], action.get("arguments", {})
        if name not in REGISTRY:
            return error("UNKNOWN_TOOL", f"unknown tool {name!r}")
        if name not in self.available_tools:
            return error("TOOL_UNAVAILABLE", f"{name} is not available for this task")
        if name in self.deprecated:
            return error("DEPRECATED", f"{name} is deprecated in environment {self.version}; "
                                       "use database.query with fields=['structure_uri']")
        message = check_arguments(name, args)
        if message:
            return error("INVALID_ARGUMENTS", message)
        for key, value in args.items():
            if key.endswith("uri") and not in_workspace(value):
                return error("PATH_OUTSIDE_WORKSPACE", f"{key} must be a workspace:// path")
        return getattr(self, "_t_" + name.replace(".", "_"))(**args)

    # -------------------------------------------------------------- artifacts
    def put(self, uri: str, kind: str, **data) -> str:
        data.update(type=kind, created_by=data.get("created_by", "tool"), step=len(self.trace) + 1)
        blob = json.dumps({k: v for k, v in data.items() if k != "step"}, sort_keys=True, default=str)
        data["checksum"] = hashlib.sha256(blob.encode()).hexdigest()[:12]
        self.artifacts[uri] = data
        return uri

    def get(self, uri: str, kinds=None):
        art = self.artifacts.get(uri) or self._lazy(uri)
        if art is None:
            return None, error("MISSING_ARTIFACT", f"{uri} does not exist")
        if kinds and art["type"] not in kinds:
            return None, error("WRONG_ARTIFACT_TYPE", f"{uri} is a {art['type']}, expected {' or '.join(kinds)}")
        return art, None

    def _lazy(self, uri: str):
        """Library structures, reference records and registry models exist without being created."""
        tail = uri.rsplit("/", 1)[-1].replace(".json", "")
        if uri.startswith("workspace://library/"):
            try:
                m = materials.by_id(tail)
            except KeyError:
                return None
            s = m.structure()
            return {"type": "structure", "created_by": "library", "structure": s.to_dict(), "material_id": m.material_id,
                    "formula": m.formula, "natoms": s.natoms, "checksum": s.checksum(), "stem": f"{m.formula}_{m.prototype}"}
        if uri.startswith("workspace://database/matact_refdb/"):
            rec = self.db_record(tail)
            return {"type": "db_record", "created_by": "matact_refdb", "record": rec, "checksum": "ref"} if rec else None
        if uri.startswith("workspace://models/registry/") and tail in self.models and self.models[tail]["kind"] == "mlp":
            return {"type": "mlp_model", "created_by": "registry", "model": tail, "pretrained": True, "checksum": tail}
        return None

    def counter(self, key: str) -> int:
        self.counters[key] = self.counters.get(key, 0) + 1
        return self.counters[key]

    def fault(self, tool: str):
        for f in self.faults:
            if f["tool"] == tool:
                return f
        return None

    def consume(self, resource: str, hours: float):
        self.resources[resource] = max(0.0, round(self.resources.get(resource, 0.0) - hours, 4))

    def public_metadata(self, uri: str, art: dict) -> dict:
        meta = {k: art[k] for k in PUBLIC_KEYS if k in art and not (k == "material_id" and art["type"] != "calc_result")}
        meta.update(uri=uri, created_by=art.get("created_by"), checksum=art.get("checksum"))
        if "content" in art:
            meta["size_bytes"] = len(art["content"])
        return meta

    # ---------------------------------------------------------- initial files
    def _materialize(self, uri: str, spec: dict):
        kind = spec["type"]
        if kind == "structure_file":
            m = materials.by_id(spec["material_id"])
            s = m.structure()
            if spec.get("supercell"):
                s = s.supercell(spec["supercell"])
            if spec.get("defect") == "duplicate_site":
                f = list(s.frac_coords[0])
                f[0] = (f[0] + 0.02 / s.abc[0]) % 1.0
                s.species.append(s.species[0])
                s.frac_coords.append(f)
            stem = uri.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            self.put(uri, "structure_file", created_by="upload", format=spec["format"],
                     content=fileio.write(s, spec["format"], stem), material_id=m.material_id)
        elif kind == "dataset":
            m = materials.by_id(spec["material_id"])
            self.put(uri, "dataset", created_by="upload", elements=m.elements, n_structures=spec["n_structures"],
                     label=spec.get("label", "PBE"), system="-".join(m.elements), material_id=m.material_id,
                     split="full", root=uri)
        elif kind == "calc_result":
            data = {k: v for k, v in spec.items() if k != "type"}
            self.put(uri, "calc_result", created_by="dft.run", **data)
        else:
            raise ValueError(f"unknown workspace spec type {kind!r}")
