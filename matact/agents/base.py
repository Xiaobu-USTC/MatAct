"""Building blocks for scripted workflows (expert demonstrations and agent logs).

A workflow is a generator that yields tool calls ``Call(name, arguments, rationale)``
and receives the resulting observation; it returns an ``Answer``. Workflows see
only the public task parameters and the observations, never reference values.
"""
from __future__ import annotations

import math
import random
from collections import namedtuple
from dataclasses import dataclass, field

from ..chem.elements import is_metallic
from ..chem.structure import parse_formula
from ..env.handlers_compute import K_DENSITY, PROTOCOL, protocol_k_mesh

Call = namedtuple("Call", "name arguments rationale")
Answer = namedtuple("Answer", "text rationale")


@dataclass
class Style:
    """Knobs that let the same workflow emulate careful experts or sloppy agents."""
    cutoff: float = 520.0
    k_scale: float = 1.0
    smearing_metals: bool = True
    split_dataset: bool = True
    verify_writes: bool = True
    epochs: int = 100
    cpu_factor: float = 1.2
    unit_slip: bool = False
    inspect_first: bool = False
    seed: int = 0
    rng: random.Random = field(default=None, repr=False)

    def __post_init__(self):
        if self.rng is None:
            self.rng = random.Random(self.seed)

    def pick(self, *options):
        return self.rng.choice(options)


class Context:
    def __init__(self, task: dict, state: dict, style: Style):
        self.params = task["params"]
        self.tools = set(task["available_tools"])
        self.deprecated = set(state.get("deprecated_tools", []))
        self.cpu = state["resources"].get("cpu_hours_remaining", 0.0)
        self.style = style


def ok(obs) -> bool:
    return bool(obs) and obs.get("status") == "success"


def ref(obs) -> str:
    return obs.get("observation_id", "the last observation")


def failure(obs, what="The task") -> Answer:
    return Answer(f"{what} could not be completed: the last call returned {obs.get('code')} "
                  f"({obs.get('message', '')}; {ref(obs)}). No result was fabricated.",
                  "The error cannot be repaired with the available tools, so report it.")


def elements_of(formula: str):
    return sorted({e for e, _ in parse_formula(formula)})


def round_up(x: float, step: float = 0.1) -> float:
    return round(math.ceil(x / step - 1e-9) * step, 4)


def retrieve(ctx: Context, formula: str, prototype: str):
    """Yield the retrieval call; return (structure uri or None, observation)."""
    if "structure.search" in ctx.deprecated and "database.query" in ctx.tools:
        obs = yield Call("database.query", {"database": "matact_refdb", "filters": {"formula": formula, "prototype": prototype},
                                            "fields": ["structure_uri"]},
                         f"structure.search is deprecated here; get the {prototype} {formula} structure URI from matact_refdb.")
        return (obs["records"][0]["structure_uri"] if ok(obs) else None), obs
    obs = yield Call("structure.search", {"formula": formula, "prototype": prototype},
                     ctx.style.pick(f"Search the structure library for {prototype} {formula}.",
                                    f"Find the {prototype} polymorph of {formula} in the library."))
    return (obs["candidates"][0]["uri"] if ok(obs) else None), obs


def report_missing_structure(ctx: Context, formula: str, prototype: str, obs, tail: str):
    alt = None
    if "structure.search" in ctx.tools and "structure.search" not in ctx.deprecated:
        alt = yield Call("structure.search", {"formula": formula},
                         f"No {prototype} {formula} was found; check which polymorphs of {formula} exist.")
    listing = [f"{c['prototype']} ({c['uri']})" for c in alt["candidates"]] if alt and ok(alt) else []
    text = (f"The structure library contains no {prototype} {formula} ({ref(obs)} returned {obs.get('code')}). " +
            (f"Available polymorphs of {formula}: {', '.join(listing)}. " if listing else
             f"No other polymorph of {formula} is available either. ") + tail)
    return Answer(text, "The requested structure does not exist, so report this instead of substituting another one.")


def dft_compute(ctx: Context, validated: dict, fields, relax: bool):
    """Prepare, run (with repairs) and extract; return (extract observation or None, last observation)."""
    style = ctx.style
    metallic = is_metallic(elements_of(validated["formula"]))
    density = K_DENSITY * style.k_scale
    args = {"structure_uri": validated["artifact_uri"], "code": "qe-7.2", "cutoff_eV": style.cutoff,
            "k_mesh": protocol_k_mesh(validated["lattice_abc_A"], density), "relax": relax}
    if metallic and style.smearing_metals:
        args["smearing"] = "mv"
    why = (f"Prepare a {'relaxation' if relax else 'single-point calculation'} with cutoff {style.cutoff:g} eV and a "
           f"{'x'.join(map(str, args['k_mesh']))} k-mesh" + (" with Marzari-Vanderbilt smearing" if "smearing" in args else "") + ".")
    prep = yield Call("dft.prepare", dict(args), why)
    if not ok(prep):
        return None, prep
    run = None
    for attempt in range(3):
        if run is None:
            limit = min(ctx.cpu, round_up(prep["estimated_cpu_hours"] * style.cpu_factor))
            why = f"Run it with a {limit:g} CPU-hour limit (estimate {prep['estimated_cpu_hours']:g} h)."
        elif run["code"] == "TIMEOUT":
            limit = min(ctx.cpu, round_up(prep["estimated_cpu_hours"] * 1.6))
            why = f"The run hit its time limit; rerun the same input with a {limit:g} CPU-hour limit."
        elif run["code"] == "NON_CONVERGENCE":
            if "smearing" in run["message"]:
                args["smearing"] = "mv"
                why = "The metallic SCF did not converge without smearing; prepare again with smearing='mv'."
            else:
                args["mixing_beta"] = 0.3
                why = "The SCF shows charge sloshing; prepare again with mixing_beta=0.3."
            prep = yield Call("dft.prepare", dict(args), why)
            if not ok(prep):
                return None, prep
            limit = min(ctx.cpu, round_up(prep["estimated_cpu_hours"] * style.cpu_factor))
            why = f"Rerun with a {limit:g} CPU-hour limit."
        else:
            break
        if limit <= 0:
            break
        run = yield Call("dft.run", {"input_uri": prep["input_uri"], "max_cpu_hours": limit}, why)
        ctx.cpu = run.get("cpu_hours_remaining", ctx.cpu)
        if ok(run):
            break
    if not ok(run):
        return None, run
    ext = yield Call("result.extract", {"output_uri": run["output_uri"], "fields": list(fields)},
                     "Extract the requested quantity from the converged run.")
    return (ext if ok(ext) else None), ext
