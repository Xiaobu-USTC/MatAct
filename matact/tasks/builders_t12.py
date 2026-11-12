"""Task builders for T1 (structure handling) and T2 (first-principles calculation)."""
from __future__ import annotations

from ..chem.structure import PROTOTYPES
from ..env import materials
from ..env.handlers_compute import reference_fixed_energy
from .common import (CODE, DFT_PROPERTIES, EXTENSION, FORMAT_NAME, FORMATS, PROTOCOL_TEXT, T1_TOOLS, T2_TOOLS,
                     budget_hours, make_task, protocol_hours, supercell_text, variants)

_ALTERNATIVES = {1: ("fcc", "bcc", "diamond"), 2: ("rocksalt", "zincblende", "wurtzite", "cscl", "fluorite")}


def missing_prototype(rng, m):
    """A prototype that does not exist in the library for the formula of ``m`` (or None)."""
    have = {x.prototype for x in materials.find(formula=m.formula)}
    options = [p for p in _ALTERNATIVES.get(len(m.elements), ()) if p not in have and p in PROTOTYPES]
    return rng.choice(options) if options else None


# ------------------------------------------------------------------------- T1
def t1_retrieve(rng, m, prototype=None):
    fmt = rng.choice(FORMATS)
    n0 = m.structure().natoms
    sc = rng.choice([[1, 1, 1], [2, 2, 2], [2, 2, 1], [3, 3, 3], [2, 1, 1]])
    if n0 * sc[0] * sc[1] * sc[2] > 400:
        sc = [1, 1, 1]
    proto = prototype or m.prototype
    sct = supercell_text(sc)
    q = variants(
        f"Retrieve the {proto} structure of {m.formula} from the structure library, validate it and write "
        f"{sct} in {FORMAT_NAME[fmt]} format.",
        f"{m.formula} ({proto}): fetch, validate, {sct} -> {FORMAT_NAME[fmt]}.",
        f"请从 structure library 中取出 {proto} {m.formula}，validate 后写成 {FORMAT_NAME[fmt]} 格式（{sct}）。")
    params = {"formula": m.formula, "prototype": proto, "format": fmt, "supercell": sc}
    if prototype:
        ref = {"check": "grounded_failure", "reason": "no_structure", "formula": m.formula, "prototype": proto}
        return make_task("structure", "retrieve_missing", "missing_data", q, params, T1_TOOLS, reference=ref,
                         budget={"cpu_hours": 1.0, "gpu_hours": 0.0}, deliverables=["grounded report"])
    ref = {"check": "structure_artifact", "material_id": m.material_id, "formula": m.formula,
           "natoms": n0 * sc[0] * sc[1] * sc[2], "format": fmt}
    return make_task("structure", "retrieve_convert", "regular", q, params, T1_TOOLS, reference=ref,
                     budget={"cpu_hours": 1.0, "gpu_hours": 0.0}, deliverables=["structure file URI", "natoms"])


def t1_file(rng, m, idx, mode="regular"):
    """mode: regular | repair (duplicate site in the upload) | wrong_path (query names a wrong extension)."""
    src = rng.choice([f for f in FORMATS if mode != "wrong_path" or f != "cif"])
    dst = rng.choice([f for f in FORMATS if f != src])
    uri = f"workspace://uploads/sample_{idx:04d}.{EXTENSION[src]}"
    spec = {"type": "structure_file", "material_id": m.material_id, "format": src}
    if mode == "repair":
        spec["defect"] = "duplicate_site"
    named = uri.rsplit(".", 1)[0] + ".cif" if mode == "wrong_path" else uri
    q = variants(f"Load the uploaded structure {named}, validate it and convert it to {FORMAT_NAME[dst]} format.",
                 f"{named}: load, validate, convert -> {FORMAT_NAME[dst]}.",
                 f"把上传的结构 {named} load 进来，validate 后 convert 成 {FORMAT_NAME[dst]}。")
    ref = {"check": "structure_artifact", "material_id": m.material_id, "formula": m.formula,
           "natoms": m.structure().natoms, "format": dst}
    subtype = {"regular": "file_convert", "repair": "repair_convert", "wrong_path": "wrong_path"}[mode]
    difficulty = "regular" if mode == "regular" else "tool_failure"
    return make_task("structure", subtype, difficulty, q, {"uri": named, "format": dst}, T1_TOOLS,
                     workspace={uri: spec}, reference=ref, budget={"cpu_hours": 1.0, "gpu_hours": 0.0},
                     deliverables=["structure file URI", "natoms"])


def t1_density(rng, m1, m2):
    d = {m.material_id: round(m.structure().density, 4) for m in (m1, m2)}
    denser = m1 if d[m1.material_id] >= d[m2.material_id] else m2
    q = variants(
        f"Retrieve {m1.prototype} {m1.formula} and {m2.prototype} {m2.formula} from the structure library, "
        "validate both and report which one has the higher density. Write the denser structure as CIF.",
        f"{m1.formula} ({m1.prototype}) vs {m2.formula} ({m2.prototype}): validate, compare density, denser -> CIF.",
        f"取出 {m1.prototype} {m1.formula} 和 {m2.prototype} {m2.formula}，分别 validate，比较 density，把密度更高的写成 CIF。")
    params = {"targets": [[m1.formula, m1.prototype], [m2.formula, m2.prototype]], "format": "cif"}
    ref = {"check": "density_compare", "material_ids": [m1.material_id, m2.material_id], "densities": d,
           "denser": denser.material_id, "formula": denser.formula, "natoms": denser.structure().natoms,
           "format": "cif"}
    return make_task("structure", "density_compare", "long_horizon", q, params, T1_TOOLS, reference=ref,
                     budget={"cpu_hours": 1.0, "gpu_hours": 0.0}, deliverables=["densities", "CIF URI"])


# ------------------------------------------------------------------------- T2
def _dft_query(phrase, unit, proto, formula):
    return variants(
        f"Using {CODE} and the MatAct {PROTOCOL_TEXT}, compute the {phrase} of {proto} {formula} from the "
        f"structure library with full relaxation. Report the value in {unit}.",
        f"{formula} ({proto}), {CODE}, Tier-B, relax: {phrase} [{unit}].",
        f"请用 {CODE} 按 Tier-B protocol（PBE, cutoff >= 520 eV, 金属加 smearing）计算 {proto} {formula} 的 "
        f"{phrase}（full relaxation），结果用 {unit} 表示。")


def t2_property(rng, m, prototype=None, fault=None, env_version="v1", budget_factor=3.0, prop=None):
    props = ["lattice_a_A", "energy_per_atom_eV"] + ([] if m.metallic else ["band_gap_eV"])
    prop = prop or rng.choice(props)
    phrase, unit, getter, tol = DFT_PROPERTIES[prop]
    proto = prototype or m.prototype
    q = _dft_query(phrase, unit, proto, m.formula)
    params = {"formula": m.formula, "prototype": proto, "property": prop, "unit": unit}
    tools = T2_TOOLS + (["database.query"] if env_version == "v2" else [])
    budget = {"cpu_hours": budget_hours(protocol_hours(m), budget_factor), "gpu_hours": 0.0}
    if prototype:
        ref = {"check": "grounded_failure", "reason": "no_structure", "formula": m.formula, "prototype": proto}
        return make_task("first_principles", "dft_missing", "missing_data", q, params, tools, budget=budget,
                         reference=ref, deliverables=["grounded report"], env_version=env_version)
    ref = {"check": "dft_value", "material_id": m.material_id, "property": prop, "value": getter(m), "unit": unit,
           "tol": tol, "protocol": {"relax": True, "metallic": m.metallic}}
    subtype = "dft_property" if not fault else {"NON_CONVERGENCE": "dft_scf_repair", "TIMEOUT": "dft_timeout"}[fault]
    return make_task("first_principles", subtype, "tool_failure" if fault else "regular", q, params, tools,
                     budget=budget, faults=[{"tool": "dft.run", "code": fault}] if fault else [], reference=ref,
                     deliverables=[prop, "output URI"], env_version=env_version)


def t2_singlepoint(rng, m, idx):
    uri = f"workspace://uploads/cell_{idx:04d}.cif"
    s = m.structure()
    natoms = s.natoms
    q = variants(
        f"Compute the total energy of the uploaded structure {uri} at fixed geometry (single-point calculation, "
        f"no relaxation) with {CODE} and the {PROTOCOL_TEXT}. Report the total energy of the cell in eV.",
        f"{uri}: single point, no relax, {CODE}, Tier-B -> total energy [eV].",
        f"对上传的 {uri} 做 single-point 计算（不 relax），用 {CODE} 和 Tier-B protocol，报告整个 cell 的 total energy（eV）。")
    ref = {"check": "dft_value", "material_id": m.material_id, "property": "total_energy_eV",
           "value": reference_fixed_energy(m, s), "unit": "eV", "tol": ("abs", 0.002 * natoms),
           "protocol": {"relax": False, "metallic": m.metallic}}
    budget = {"cpu_hours": budget_hours(protocol_hours(m, relax=False)), "gpu_hours": 0.0}
    return make_task("first_principles", "dft_singlepoint", "regular", q, {"uri": uri, "property": "total_energy_eV",
                     "unit": "eV"}, T2_TOOLS, workspace={uri: {"type": "structure_file", "material_id": m.material_id,
                                                                "format": "cif"}},
                     budget=budget, reference=ref, deliverables=["total_energy_eV", "output URI"])


def t2_mismatch(rng, m1, m2):
    a1, a2 = m1.a_ref, m2.a_ref
    q = variants(
        f"Using {CODE} and the {PROTOCOL_TEXT}, compute the relaxed lattice constants of {m1.prototype} "
        f"{m1.formula} (a1) and {m2.prototype} {m2.formula} (a2) and report the lattice mismatch (a2 - a1)/a1 in percent.",
        f"Tier-B relax: a({m1.formula}, {m1.prototype}), a({m2.formula}, {m2.prototype}) -> mismatch (a2-a1)/a1 [%].",
        f"用 {CODE} 和 Tier-B protocol 分别 relax {m1.prototype} {m1.formula}（a1）和 {m2.prototype} {m2.formula}（a2），"
        "报告 lattice mismatch (a2 - a1)/a1（%）。")
    params = {"targets": [[m1.formula, m1.prototype], [m2.formula, m2.prototype]], "property": "lattice_a_A"}
    ref = {"check": "dft_mismatch", "material_ids": [m1.material_id, m2.material_id], "values": [a1, a2],
           "value": round((a2 - a1) / a1 * 100, 4), "tol": 0.15, "unit": "%",
           "protocol": {"relax": True, "metallic": False}}
    budget = {"cpu_hours": budget_hours(protocol_hours(m1) + protocol_hours(m2)), "gpu_hours": 0.0}
    return make_task("first_principles", "dft_mismatch", "long_horizon", q, params, T2_TOOLS, budget=budget,
                     reference=ref, deliverables=["a1", "a2", "mismatch (%)"])
