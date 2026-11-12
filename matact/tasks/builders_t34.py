"""Task builders for T3 (ML potentials), T4 (databases) and composite OOD-comp workflows."""
from __future__ import annotations

import math

from ..env import materials
from .common import (CODE, COMPOSITE_TOOLS, DB_FIELDS, DB_UNIT, DFT_PROPERTIES, PROTOCOL_TEXT, T3_TOOLS, T4_TOOLS,
                     budget_hours, make_task, protocol_hours, variants)

EPOCHS = 100


def _dataset(m, idx, n):
    system = "-".join(m.elements)
    uri = f"workspace://datasets/{system}_pbe_{idx:04d}.extxyz"
    return system, uri, {"type": "dataset", "material_id": m.material_id, "n_structures": n, "label": "PBE"}


def expected_pretrained(elements, max_cost, env_version="v1"):
    """Cheapest compatible registry potential, mirroring mlp.select."""
    from ..env.environment import ScientificEnvironment
    from .common import MODELS_V1, MODELS_V2
    env = ScientificEnvironment({"env_version": env_version, "available_tools": ["mlp.select"], "initial_state": {
        "available_models": MODELS_V2 if env_version == "v2" else MODELS_V1, "resource_budget": {}}})
    obs = env.step({"name": "mlp.select", "arguments": {"task": "energy_forces", "elements": elements,
                                                        "max_cost": max_cost}})
    return obs.get("model")


# ------------------------------------------------------------------------- T3
def t3_finetune(rng, m, idx, fault=False, compare=False, missing=False, other=None):
    n = rng.choice([200, 300, 400, 600])
    frac = rng.choice([0.1, 0.15, 0.2])
    system, uri, spec = _dataset(m, idx, n)
    workspace = {} if missing else {uri: spec}
    if missing and other is not None:
        _, other_uri, other_spec = _dataset(other, idx + 5000, n)
        workspace[other_uri] = other_spec
    pct = int(round(frac * 100))
    extra = (" Also evaluate the pretrained base model on the same held-out set and state whether fine-tuning "
             "improved the energy MAE.") if compare else ""
    q = variants(
        f"Fine-tune a machine-learning interatomic potential on the labelled {system} dataset {uri}: hold out "
        f"{pct}% of the structures as a test set (seed 0), train on the rest and report the energy and force MAE "
        f"on the held-out set.{extra}",
        f"{uri}: {pct}% hold-out split, fine-tune MLP, report energy/force MAE on test{', compare with base' if compare else ''}.",
        f"在 {system} 数据集 {uri} 上 fine-tune 一个 ML potential：留出 {pct}% 作为 test set（seed 0），"
        f"在其余数据上训练并报告 test set 上的 energy/force MAE{'，并与 pretrained base model 比较' if compare else ''}。")
    n_train = n - max(1, round(n * frac))
    budget = {"cpu_hours": 0.0, "gpu_hours": budget_hours(n_train * EPOCHS * 5e-5 + 0.3, 2.5, 1.0)}
    params = {"dataset_uri": uri, "test_fraction": frac, "compare": compare}
    if missing:
        ref = {"check": "grounded_failure", "reason": "no_dataset", "dataset_uri": uri}
        return make_task("ml_potential", "mlp_missing", "missing_data", q, params, T3_TOOLS, workspace=workspace,
                         budget=budget, reference=ref, deliverables=["grounded report"])
    ref = {"check": "mlp_compare" if compare else "mlp_finetune", "dataset_uri": uri, "test_fraction": frac,
           "material_id": m.material_id}
    subtype = "mlp_compare" if compare else ("mlp_unstable_repair" if fault else "mlp_finetune")
    difficulty = "long_horizon" if compare else ("tool_failure" if fault else "regular")
    faults = [{"tool": "mlp.train", "code": "UNSTABLE_TRAINING"}] if fault else []
    return make_task("ml_potential", subtype, difficulty, q, params, T3_TOOLS, workspace=workspace, budget=budget,
                     faults=faults, reference=ref, deliverables=["energy MAE", "force MAE", "model URI"])


def t3_pretrained(rng, m, idx, env_version="v1"):
    n = rng.choice([200, 300, 400])
    system, uri, spec = _dataset(m, idx, n)
    max_cost = 4
    q = variants(
        f"Select the cheapest registered pretrained ML potential (cost <= {max_cost}) that supports the labelled "
        f"{system} dataset {uri} and report its energy and force MAE on the full dataset without fine-tuning.",
        f"{uri}: cheapest compatible pretrained MLP (cost <= {max_cost}), no fine-tuning, energy/force MAE.",
        f"为 {system} 数据集 {uri} 选择最便宜的兼容 pretrained ML potential（cost <= {max_cost}），不做 fine-tuning，"
        "报告 energy/force MAE。")
    ref = {"check": "mlp_pretrained", "dataset_uri": uri, "model": expected_pretrained(m.elements, max_cost, env_version)}
    return make_task("ml_potential", "mlp_pretrained", "regular", q, {"dataset_uri": uri, "max_cost": max_cost},
                     T3_TOOLS, workspace={uri: spec}, budget={"cpu_hours": 0.0, "gpu_hours": 1.0}, reference=ref,
                     deliverables=["model", "energy MAE", "force MAE"], env_version=env_version)


# ------------------------------------------------------------------------- T4
def t4_lookup(rng, m, env_version="v1"):
    prop = rng.choice(sorted(DB_FIELDS))
    phrase, unit = DB_FIELDS[prop]
    if unit == "eV/atom" and rng.random() < 0.5:
        unit = "meV/atom"
    q = variants(
        f"Look up the {phrase} of {m.prototype} {m.formula} in the MatAct reference database (matact_refdb) and "
        f"report it in {unit} together with the record URI.",
        f"matact_refdb: {phrase} of {m.formula} ({m.prototype}) [{unit}] + record URI.",
        f"在 matact_refdb 里查 {m.prototype} {m.formula} 的 {phrase}，用 {unit} 报告，并给出 record URI。")
    params = {"formula": m.formula, "prototype": m.prototype, "property": prop, "unit": unit}
    covered = env_version == "v2" or m.in_db_v1
    if not covered:
        ref = {"check": "grounded_failure", "reason": "no_record", "material_id": m.material_id}
        return make_task("database", "db_missing", "missing_data", q, params, T4_TOOLS, reference=ref,
                         deliverables=["grounded report"])
    ref = {"check": "db_value", "material_id": m.material_id, "property": prop,
           "value": m.record()[prop], "unit": unit, "tol_rel": 0.01}
    return make_task("database", "db_lookup", "regular", q, params, T4_TOOLS, reference=ref,
                     deliverables=[prop, "record URI"], env_version=env_version)


def filter_candidates(allowed_splits):
    """(prototype, element, lo, hi, ids) windows whose matching v1 records all lie in ``allowed_splits``."""
    recs = [m for m in materials.library() if m.in_db_v1 and not m.metallic]
    out = []
    for proto in sorted({m.prototype for m in recs}):
        for el in sorted({e for m in recs if m.prototype == proto for e in m.elements}):
            group = sorted((m for m in recs if m.prototype == proto and el in m.elements), key=lambda x: x.band_gap)
            for i in range(len(group)):
                for j in range(i, min(i + 3, len(group))):
                    lo = math.floor(group[i].band_gap * 10) / 10
                    hi = math.ceil(group[j].band_gap * 10) / 10
                    hits = [m for m in group if lo <= m.band_gap <= hi]
                    if hits and all(m.split in allowed_splits for m in hits):
                        out.append((proto, el, lo, hi, tuple(sorted(m.material_id for m in hits))))
    return sorted(set(out))


def t4_filter(rng, candidates):
    proto, el, lo, hi, ids = rng.choice(candidates)
    q = variants(
        f"List the material IDs of all {proto} compounds in matact_refdb that contain {el} and have a band gap "
        f"between {lo:.1f} and {hi:.1f} eV.",
        f"matact_refdb: {proto}, contains {el}, {lo:.1f} <= gap <= {hi:.1f} eV -> material IDs.",
        f"在 matact_refdb 中列出所有含 {el}、band gap 在 {lo:.1f}–{hi:.1f} eV 之间的 {proto} 化合物的 material ID。")
    params = {"prototype": proto, "element": el, "band_gap_min": lo, "band_gap_max": hi}
    return make_task("database", "db_filter", "regular", q, params, T4_TOOLS,
                     reference={"check": "db_ids", "ids": list(ids)}, deliverables=["material IDs"])


def t4_insert(rng, ms, idx, fault=False):
    workspace, records = {}, []
    for k, m in enumerate(ms, start=1):
        prop = rng.choice(sorted(DB_UNIT))
        uri = f"workspace://results/calc_{idx:04d}_{k}.json"
        value = m.record()[prop]
        workspace[uri] = {"type": "calc_result", "material_id": m.material_id, "formula": m.formula,
                          "property": prop, "value": value, "unit": DB_UNIT[prop], "code": CODE,
                          "calc_type": "PBE/relaxed (MatAct Tier-B protocol)"}
        records.append({"material_id": m.material_id, "property": prop, "value": value, "unit": DB_UNIT[prop]})
    uris = sorted(workspace)
    what = f"the computed result in {uris[0]}" if len(uris) == 1 else "the computed results in " + " and ".join(uris)
    q = variants(f"Register {what} in the user database (matact_userdb) and verify that the record"
                 f"{'s were' if len(uris) > 1 else ' was'} written correctly.",
                 f"{', '.join(uris)} -> matact_userdb insert + verify.",
                 f"把 {'、'.join(uris)} 中的计算结果写入 matact_userdb，并 verify 写入是否正确。")
    subtype = "db_insert_repair" if fault else ("db_insert_two" if len(ms) > 1 else "db_insert")
    difficulty = "tool_failure" if fault else ("long_horizon" if len(ms) > 1 else "regular")
    return make_task("database", subtype, difficulty, q, {"result_uris": uris}, T4_TOOLS, workspace=workspace,
                     faults=[{"tool": "database.update", "code": "SCHEMA_CONFLICT"}] if fault else [],
                     reference={"check": "db_write", "records": records}, deliverables=["receipt URI", "record URI"])


# ------------------------------------------------------------------ OOD-comp
def c_compute_register(rng, m):
    props = ["lattice_a_A", "energy_per_atom_eV"] + ([] if m.metallic else ["band_gap_eV"])
    prop = rng.choice(props)
    phrase, unit, getter, tol = DFT_PROPERTIES[prop]
    q = variants(
        f"Using {CODE} and the {PROTOCOL_TEXT}, compute the {phrase} of {m.prototype} {m.formula} with full "
        "relaxation, then register the value in the user database (matact_userdb) with the DFT output as "
        "source_uri and verify the write.",
        f"{m.formula} ({m.prototype}): Tier-B {phrase} -> matact_userdb (source = DFT output), verify.",
        f"用 {CODE} 和 Tier-B protocol 计算 {m.prototype} {m.formula} 的 {phrase}（full relaxation），再把结果写入 "
        "matact_userdb（source_uri 用 DFT output）并 verify。")
    ref = {"check": "dft_write", "material_id": m.material_id, "property": prop, "value": getter(m), "unit": unit,
           "tol": tol, "protocol": {"relax": True, "metallic": m.metallic}}
    return make_task("composite", "compute_register", "long_horizon", q,
                     {"formula": m.formula, "prototype": m.prototype, "property": prop, "unit": unit},
                     COMPOSITE_TOOLS, budget={"cpu_hours": budget_hours(protocol_hours(m)), "gpu_hours": 0.0},
                     reference=ref, deliverables=[prop, "record URI"])


def c_db_fallback(rng, m):
    props = ["lattice_a_A", "energy_per_atom_eV"] + ([] if m.metallic else ["band_gap_eV"])
    prop = rng.choice(props)
    phrase, unit, getter, tol = DFT_PROPERTIES[prop]
    q = variants(
        f"Report the {phrase} of {m.prototype} {m.formula} in {unit}. Use the record in matact_refdb if it exists; "
        f"otherwise compute it with {CODE} and the {PROTOCOL_TEXT} (full relaxation).",
        f"{phrase} of {m.formula} ({m.prototype}) [{unit}]: matact_refdb first, else Tier-B DFT.",
        f"报告 {m.prototype} {m.formula} 的 {phrase}（{unit}）：优先查 matact_refdb，没有记录就用 {CODE} 按 Tier-B protocol 计算。")
    ref = {"check": "dft_value", "material_id": m.material_id, "property": prop, "value": getter(m), "unit": unit,
           "tol": tol, "protocol": {"relax": True, "metallic": m.metallic}}
    return make_task("composite", "db_fallback_dft", "long_horizon", q,
                     {"formula": m.formula, "prototype": m.prototype, "property": prop, "unit": unit},
                     COMPOSITE_TOOLS, budget={"cpu_hours": budget_hours(protocol_hours(m)), "gpu_hours": 0.0},
                     reference=ref, deliverables=[prop, "source URI"])
