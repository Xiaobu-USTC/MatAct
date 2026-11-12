"""Category-specific task-success checks (Supplementary Note 7) run against the final environment state."""
from __future__ import annotations

import re

from ..chem.structure import Structure
from ..env import materials
from ..env.handlers_compute import K_DENSITY, PROTOCOL, k_density
from .answer import citations, numbers

_ACCEPT_UNITS = {"Å": ("Å", ""), "A": ("Å", ""), "eV/atom": ("eV/atom",), "meV/atom": ("meV/atom",),
                 "eV": ("eV",), "%": ("%",)}
_ACK = re.compile(r"\b(no|not|none|missing|does not exist|unavailable|absent|cannot|could not)\b|没有|不存在|缺失|无法",
                  re.I)


def within(value, target, tol) -> bool:
    kind, size = tol
    limit = size * abs(target) if kind == "rel" else size
    return abs(value - target) <= max(limit, 1e-6)


def reported(answer, target, unit, tol) -> bool:
    units = _ACCEPT_UNITS.get(unit, (unit,))
    return any(u in units and within(v, target, tol) for v, u, _ in numbers(answer))


def protocol_ok(art: dict, protocol: dict) -> bool:
    st = art["settings"]
    m = materials.by_id(art["material_id"])
    return (st["cutoff_eV"] >= PROTOCOL["cutoff_eV"] and k_density(art["abc"], st["k_mesh"]) >= K_DENSITY - 1e-6
            and st["relax"] == protocol["relax"] and (not m.metallic or st["smearing"] != "none"))


def _arts(env, kind, **match):
    return [(u, a) for u, a in env.artifacts.items() if a["type"] == kind and all(a.get(k) == v for k, v in match.items())]


def _calls(steps, name, ok=True):
    return [s for s in steps if (s.get("action") or {}).get("name") == name
            and (not ok or s["observation"].get("status") == "success")]


def _dft_outputs(env, uris, material_id, protocol):
    cited = [a for u, a in _arts(env, "dft_output", material_id=material_id) if u in uris]
    return bool(cited), bool(cited) and all(protocol_ok(a, protocol) for a in cited)


def _verified_after_write(steps) -> bool:
    last = max((i for i, s in enumerate(steps) if (s.get("action") or {}).get("name") == "database.update"
                and s["observation"].get("status") == "success"), default=None)
    return last is not None and any((s.get("action") or {}).get("name") == "database.query"
                                    and s["action"]["arguments"].get("database") == "matact_userdb"
                                    and s["observation"].get("status") == "success" for s in steps[last + 1:])


def check_task(task: dict, steps: list, answer: str, env) -> dict:
    """Return {"checks": {name: bool}, "science": [bool], "value_checks": [names]}."""
    ref, answer = task["reference"], answer or ""
    uris, obs_ids = citations(answer)
    kind = ref["check"]
    checks, science, value_checks = {}, [], []
    if kind == "structure_artifact":
        good = [(u, a) for u, a in _arts(env, "structure", created_by="structure.convert", material_id=ref["material_id"])
                if a.get("format") == ref["format"] and a["natoms"] == ref["natoms"] and a["formula"] == ref["formula"]]
        cited = [a for u, a in good if u in uris]
        geometry = bool(cited) and all(Structure.from_dict(a["structure"]).closest_pair()[0] >= 0.5 for a in cited)
        checks = {"artifact": bool(good), "cited": bool(cited), "geometry": geometry}
        science = [geometry]
    elif kind == "density_compare":
        validated = {a.get("material_id") for _, a in _arts(env, "validated_structure")}
        cifs = [(u, a) for u, a in _arts(env, "structure", created_by="structure.convert", format="cif") if u in uris]
        values = [v for v, _, _ in numbers(answer)]
        checks = {"validated_both": set(ref["material_ids"]) <= validated,
                  "denser_cif": bool(cifs) and all(a["material_id"] == ref["denser"] for _, a in cifs),
                  "densities": all(any(abs(v - d) <= 0.011 for v in values) for d in ref["densities"].values())}
        value_checks = ["densities"]
    elif kind in ("dft_value", "dft_mismatch"):
        mids = ref.get("material_ids") or [ref["material_id"]]
        results = [_dft_outputs(env, uris, mid, ref["protocol"]) for mid in mids]
        tol = ("abs", ref["tol"]) if kind == "dft_mismatch" else tuple(ref["tol"])
        checks = {"value": reported(answer, ref["value"], ref["unit"], tol),
                  "cited_output": all(c for c, _ in results), "protocol": all(p for _, p in results)}
        science, value_checks = [checks["protocol"]], ["value"]
    elif kind in ("mlp_finetune", "mlp_compare"):
        root = ref["dataset_uri"]
        n = task["initial_state"]["workspace"][root]["n_structures"]
        n_test = max(1, round(n * ref["test_fraction"]))
        tests = {u for u, a in _arts(env, "dataset", root=root, split="test") if a["n_structures"] == n_test}
        models = {u for u, a in _arts(env, "mlp_model", created_by="mlp.train", root=root, train_split="train")}
        evals = [s for s in _calls(steps, "mlp.evaluate") if s["action"]["arguments"]["model_uri"] in models
                 and s["action"]["arguments"]["test_uri"] in tests]
        values = [v for v, _, _ in numbers(answer)]
        rep = evals and any(any(abs(v - s["observation"]["energy_mae_meV_atom"]) < 0.006 for v in values) and
                            any(abs(v - s["observation"]["force_mae_meV_A"]) < 0.06 for v in values) for s in evals)
        checks = {"held_out_split": bool(tests), "trained_on_train_split": bool(models), "evaluated": bool(evals),
                  "no_overlap": bool(evals) and not any(s["observation"]["warnings"] for s in evals),
                  "reported": bool(rep), "cited": bool(uris & (models | {s["observation"]["report_uri"] for s in evals}))}
        if kind == "mlp_compare":
            base = [s for s in _calls(steps, "mlp.evaluate") if "/models/registry/" in s["action"]["arguments"]["model_uri"]
                    and s["action"]["arguments"]["test_uri"] in tests]
            ok_cmp = bool(base and evals)
            if ok_cmp:
                improved = evals[-1]["observation"]["energy_mae_meV_atom"] < base[-1]["observation"]["energy_mae_meV_atom"]
                negated = re.search(r"did not improve|not improved|no improvement", answer, re.I)
                ok_cmp = ("improv" in answer.lower()) and (bool(negated) != improved)
            checks["compared_with_base"] = ok_cmp
        science, value_checks = [checks["held_out_split"] and checks["no_overlap"]], ["reported"]
    elif kind == "mlp_pretrained":
        target = f"workspace://models/registry/{ref['model']}"
        evals = [s for s in _calls(steps, "mlp.evaluate") if s["action"]["arguments"]["model_uri"] == target
                 and s["action"]["arguments"]["test_uri"] == ref["dataset_uri"]]
        values = [v for v, _, _ in numbers(answer)]
        rep = any(any(abs(v - s["observation"]["energy_mae_meV_atom"]) < 0.006 for v in values) for s in evals)
        checks = {"model": bool(evals) and ref["model"] in answer, "reported": rep}
        value_checks = ["reported"]
    elif kind == "db_value":
        scale = 1000.0 if ref["unit"] == "meV/atom" else 1.0
        tol = ("rel", ref["tol_rel"]) if ref["value"] else ("abs", 1e-3)
        record = f"workspace://database/matact_refdb/{ref['material_id']}"
        checks = {"value": reported(answer, ref["value"] * scale, ref["unit"], tol), "cited": record in uris}
        science, value_checks = [checks["value"]], ["value"]
    elif kind == "db_ids":
        checks = {"ids": set(re.findall(r"\bmx-\d{4}\b", answer)) == set(ref["ids"])}
        value_checks = ["ids"]
    elif kind in ("db_write", "dft_write"):
        records = ref.get("records") or [{"material_id": ref["material_id"], "property": ref["property"]}]
        written, units_ok, proto = True, True, True
        cited_set = set()
        for rec in records:
            row = env.userdb.get(f"{rec['material_id']}/{rec['property']}")
            if row is None:
                written = False
                continue
            cited_set.add(row["record_uri"])
            if kind == "db_write":
                written &= abs(row["value"] - rec["value"]) <= 1e-9 + 1e-6 * abs(rec["value"]) and row["unit"] == rec["unit"]
            else:
                written &= within(row["value"], ref["value"], tuple(ref["tol"]))
                src = env.artifacts.get(row["source_uri"], {})
                proto &= src.get("type") == "dft_output" and src.get("material_id") == ref["material_id"] \
                    and protocol_ok(src, ref["protocol"])
        receipts = {u for u, _ in _arts(env, "db_receipt")}
        checks = {"written": written, "verified": _verified_after_write(steps), "cited": bool(uris & (receipts | cited_set))}
        if kind == "dft_write":
            checks["protocol"] = proto
            science = [proto]
        value_checks = ["written"]
    elif kind == "grounded_failure":
        substituted = {"no_structure": ("dft.run", "structure.convert"), "no_record": ("database.update",),
                       "no_dataset": ("mlp.train", "mlp.evaluate")}[ref["reason"]]
        checks = {"acknowledged": bool(_ACK.search(answer)), "cited": bool(uris or obs_ids),
                  "no_substitution": not any(_calls(steps, name) for name in substituted)}
        value_checks = ["acknowledged"]
    else:
        raise ValueError(f"unknown check {kind!r}")
    return {"checks": checks, "science": science, "value_checks": value_checks}
