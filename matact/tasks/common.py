"""Shared constants and helpers for MatAct-Bench task construction."""
from __future__ import annotations

import math

from ..env.handlers_compute import PROTOCOL, estimate_cpu_hours, protocol_k_mesh

CODE = "qe-7.2"
MODELS_V1 = ["qe-7.2", "chgnet-0.3", "mace-mp-medium"]
MODELS_V2 = MODELS_V1 + ["mace-ft-matact-v2"]
PROTOCOL_TEXT = ("Tier-B protocol (PBE, plane-wave cutoff >= 520 eV, k-point spacing <= 0.15 Å^-1, "
                 "Marzari-Vanderbilt smearing for metals)")
FORMATS = ("cif", "poscar", "xyz")
FORMAT_NAME = {"cif": "CIF", "poscar": "POSCAR", "xyz": "extended XYZ"}
EXTENSION = {"cif": "cif", "poscar": "vasp", "xyz": "xyz"}

T1_TOOLS = ["structure.search", "structure.load", "structure.validate", "structure.convert", "artifact.inspect"]
T2_TOOLS = ["structure.search", "structure.load", "structure.validate", "dft.prepare", "dft.run",
            "result.extract", "artifact.inspect"]
T3_TOOLS = ["artifact.inspect", "code.execute", "mlp.select", "mlp.train", "mlp.evaluate"]
T4_TOOLS = ["database.query", "database.update", "artifact.inspect", "code.execute"]
COMPOSITE_TOOLS = sorted(set(T2_TOOLS + T4_TOOLS))

# property: (phrase, unit, reference getter, tolerance)
DFT_PROPERTIES = {
    "lattice_a_A": ("relaxed lattice constant a", "Å", lambda m: m.a_ref, ("rel", 0.005)),
    "energy_per_atom_eV": ("relaxed total energy per atom", "eV/atom", lambda m: m.energy_per_atom, ("abs", 0.002)),
    "band_gap_eV": ("PBE band gap", "eV", lambda m: m.band_gap, ("abs", 0.05)),
}
DB_FIELDS = {
    "band_gap_eV": ("PBE band gap", "eV"),
    "formation_energy_eV_per_atom": ("formation energy", "eV/atom"),
    "energy_per_atom_eV": ("total energy per atom", "eV/atom"),
    "lattice_a_A": ("relaxed lattice constant a", "Å"),
}
DB_UNIT = {"band_gap_eV": "eV", "formation_energy_eV_per_atom": "eV/atom", "energy_per_atom_eV": "eV/atom",
           "lattice_a_A": "A"}


def protocol_hours(m, relax=True, structure=None) -> float:
    s = structure or m.structure()
    k = protocol_k_mesh(s.abc)
    return estimate_cpu_hours(s.natoms, k, PROTOCOL["cutoff_eV"], relax, m.metallic)


def budget_hours(hours: float, factor: float = 3.0, minimum: float = 2.0) -> float:
    return max(minimum, math.ceil(hours * factor * 2) / 2)


def variants(formal: str, terse: str, mixed: str) -> dict:
    novice = ("Hi, I am new to computational materials science. " + formal +
              " Please explain briefly what you did and give the final numbers with units.")
    return {"formal": formal, "terse": terse, "novice": novice, "mixed": mixed}


def supercell_text(sc) -> str:
    return "the conventional cell" if list(sc) == [1, 1, 1] else "a {}x{}x{} supercell".format(*sc)


def make_task(category, subtype, difficulty, queries, params, tools, workspace=None, budget=None,
              faults=None, reference=None, deliverables=None, models=None, env_version="v1") -> dict:
    return {
        "category": category, "subtype": subtype, "difficulty": difficulty, "env_version": env_version,
        "query": queries["formal"], "query_variants": queries, "params": params,
        "initial_state": {"workspace": workspace or {},
                          "available_models": models or (MODELS_V2 if env_version == "v2" else MODELS_V1),
                          "resource_budget": budget or {"cpu_hours": 0.0, "gpu_hours": 0.0}},
        "available_tools": list(tools),
        "success_spec": {"deliverables": deliverables or [], "must_be_grounded": True},
        "faults": faults or [],
        "reference": reference or {},
    }


def public_view(task: dict) -> dict:
    """The part of a task that a policy may see (no faults, references or hidden file specs)."""
    return {"task_id": task["task_id"], "query": task["query"], "available_tools": task["available_tools"],
            "resource_budget": task["initial_state"]["resource_budget"],
            "success_spec": task["success_spec"]}
