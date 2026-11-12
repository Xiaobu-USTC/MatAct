"""Handlers for first-principles and machine-learning-potential tools.

The DFT executor evaluates the requested quantity at the requested settings: the
returned values equal the reference-protocol values plus basis-set,
k-point and strain errors that depend on the requested settings. The ML
potential executor models accuracy as a function of data size, epochs and the
chemical system. Runtime and resource use follow simple scaling laws.
"""
from __future__ import annotations

import math

from ..chem.elements import ELEMENTS
from ..chem.structure import Structure
from . import materials
from .common import basename, error
from .materials import unit_hash

PROTOCOL = {"cutoff_eV": 520.0, "k_spacing_A_inv": 0.15}
# k-point density n_i * |a_i| (Angstrom) implied by a reciprocal-space spacing of 0.15 1/Angstrom.
K_DENSITY = 2 * math.pi / PROTOCOL["k_spacing_A_inv"]


def k_density(abc, k_mesh) -> float:
    """Smallest product n_i * |a_i| (Angstrom); larger means a denser k-point mesh."""
    return min(n * length for n, length in zip(k_mesh, abc))


def protocol_k_mesh(abc, density=None):
    """Smallest k-point mesh that meets the Tier-B k-point spacing."""
    d = density or K_DENSITY
    return [max(1, math.ceil(d / length - 1e-9)) for length in abc]


def reference_fixed_energy(material, s: Structure) -> float:
    """Converged total energy (eV) of ``s`` at fixed geometry."""
    ref = material.structure(relaxed=True)
    strain = ((s.volume / s.natoms) / (ref.volume / ref.natoms)) ** (1 / 3) - 1
    return round((material.energy_per_atom + 5.0 * strain ** 2) * s.natoms, 5)


def estimate_cpu_hours(natoms, k_mesh, cutoff, relax, metallic) -> float:
    nk = k_mesh[0] * k_mesh[1] * k_mesh[2]
    t = 2.5e-4 * natoms ** 2 * nk ** 0.8 * (cutoff / 400.0) ** 1.5 * (2.0 if relax else 1.0) * (1.3 if metallic else 1.0)
    return round(max(t, 0.05), 2)


def system_factor(elements) -> float:
    return 1.0 + 0.5 * unit_hash("-".join(sorted(elements)), "sys") + 0.25 * (len(set(elements)) - 1)


def dft_result(material, s: Structure, cutoff, k_mesh, relax) -> dict:
    """DFT result for structure ``s`` (derived from ``material``)."""
    hardness = max(ELEMENTS[e].hardness for e in s.species)
    err_cut = 30.0 * math.exp(-(cutoff - 250.0) / 60.0) * hardness
    kd = k_density(s.abc, k_mesh)
    err_k = 120.0 * math.exp(-kd / 9.0) if material.metallic else 50.0 * math.exp(-kd / 6.0)
    sign = 1.0 if unit_hash(material.material_id, tuple(k_mesh), "ksign") < 0.5 else -1.0
    ref = material.structure(relaxed=True)
    strain = 0.0 if relax else ((s.volume / s.natoms) / (ref.volume / ref.natoms)) ** (1 / 3) - 1
    e_atom = material.energy_per_atom + (err_cut + sign * err_k + 5000.0 * strain ** 2) / 1000.0
    mult = [max(1, round(x / y)) for x, y in zip(s.abc, ref.abc)]
    if relax:
        lat = [material.a_ref * mult[0] * (1 + 3e-4 * err_cut), (material.c_ref or material.a_ref) * mult[2]]
    else:
        lat = [s.abc[0], s.abc[2]]
    gap = 0.0 if material.metallic else max(0.0, material.band_gap * (1 - 2.0 * strain) + 0.01 * err_k)
    out = {"total_energy_eV": round(e_atom * s.natoms, 5), "energy_per_atom_eV": round(e_atom, 5),
           "lattice_a_A": round(lat[0], 4), "band_gap_eV": round(gap, 4),
           "max_force_eV_A": 0.0004 if relax else 0.0, "pressure_kbar": round(0.3 if relax else -3000.0 * strain, 2),
           "converged": True}
    if material.prototype == "wurtzite":
        out["lattice_c_A"] = round(lat[1] * (1 + 3e-4 * err_cut) if relax else lat[1], 4)
    return out


class ComputeHandlers:
    # ------------------------------------------------------------------ DFT
    def _t_dft_prepare(self, structure_uri, code, cutoff_eV, k_mesh, relax, smearing="none", mixing_beta=0.7):
        art, err = self.get(structure_uri, ("validated_structure",))
        if err:
            if err["code"] == "WRONG_ARTIFACT_TYPE":
                err["message"] += "; validate the structure with structure.validate first"
            return err
        if code not in self.models or self.models[code]["kind"] != "dft":
            return error("MODEL_UNAVAILABLE", f"{code} is not registered",
                         registered=[n for n, m in self.models.items() if m["kind"] == "dft"])
        if not 150 <= cutoff_eV <= 1500:
            return error("INVALID_PARAMETER", "cutoff_eV must be between 150 and 1500")
        if any(k < 1 or k > 24 for k in k_mesh):
            return error("INVALID_PARAMETER", "k_mesh entries must be between 1 and 24")
        if not 0 < mixing_beta <= 1:
            return error("INVALID_PARAMETER", "mixing_beta must be in (0, 1]")
        s = Structure.from_dict(art["structure"])
        m = materials.by_id(art["material_id"])
        est = estimate_cpu_hours(s.natoms, k_mesh, cutoff_eV, relax, m.metallic)
        run = f"workspace://runs/{art['stem']}-{self.counter('run:' + art['stem'])}/input"
        self.put(run, "dft_input", created_by="dft.prepare", structure=art["structure"], material_id=m.material_id,
                 code=code, cutoff_eV=cutoff_eV, k_mesh=list(k_mesh), relax=relax, smearing=smearing,
                 mixing_beta=mixing_beta, estimate=est, stem=art["stem"], natoms=s.natoms)
        return {"status": "success", "input_uri": run, "natoms": s.natoms,
                "n_kpoints": k_mesh[0] * k_mesh[1] * k_mesh[2], "estimated_cpu_hours": est}

    def _t_dft_run(self, input_uri, max_cpu_hours):
        art, err = self.get(input_uri, ("dft_input",))
        if err:
            return err
        remaining = self.resources.get("cpu_hours", 0.0)
        if max_cpu_hours <= 0:
            return error("INVALID_ARGUMENTS", "max_cpu_hours must be positive")
        if max_cpu_hours > remaining + 1e-9:
            return error("BUDGET_EXCEEDED", f"max_cpu_hours={max_cpu_hours} exceeds the remaining CPU budget",
                         cpu_hours_remaining=round(remaining, 2))
        m = materials.by_id(art["material_id"])
        need = art["estimate"]
        fault = self.fault("dft.run")
        if fault and fault["code"] == "TIMEOUT":
            need = round(need * 1.4, 2)
        if need > max_cpu_hours:
            self.consume("cpu_hours", max_cpu_hours)
            return error("TIMEOUT", f"run killed at the {max_cpu_hours} CPU-hour limit before completion",
                         cpu_hours_used=max_cpu_hours, cpu_hours_remaining=round(self.resources["cpu_hours"], 2))
        if (m.metallic and art["smearing"] == "none") or \
                (fault and fault["code"] == "NON_CONVERGENCE" and art["mixing_beta"] > 0.3):
            used = round(need * 0.5, 2)
            self.consume("cpu_hours", used)
            hint = "metallic system without smearing; use smearing='mv'" if art["smearing"] == "none" and m.metallic \
                else "charge sloshing; reduce mixing_beta (e.g. 0.3)"
            return error("NON_CONVERGENCE", f"SCF did not converge in 100 iterations ({hint})",
                         cpu_hours_used=used, cpu_hours_remaining=round(self.resources["cpu_hours"], 2))
        self.consume("cpu_hours", need)
        s = Structure.from_dict(art["structure"])
        results = dft_result(m, s, art["cutoff_eV"], art["k_mesh"], art["relax"])
        out = input_uri.rsplit("/", 1)[0] + "/output"
        self.put(out, "dft_output", created_by="dft.run", results=results, material_id=m.material_id,
                 settings={k: art[k] for k in ("code", "cutoff_eV", "k_mesh", "relax", "smearing")},
                 abc=list(s.abc), natoms=s.natoms)
        return {"status": "success", "converged": True, "output_uri": out, "cpu_hours_used": need,
                "cpu_hours_remaining": round(self.resources["cpu_hours"], 2)}

    def _t_result_extract(self, output_uri, fields):
        art, err = self.get(output_uri, ("dft_output",))
        if err:
            return err
        missing = [f for f in fields if f not in art["results"]]
        if missing or not fields:
            return error("MISSING_FIELD", f"field(s) not available: {missing or 'none requested'}",
                         available=sorted(art["results"]))
        return {"status": "success", **{f: art["results"][f] for f in fields}, "source_uri": output_uri}

    # --------------------------------------------------------- ML potentials
    def _supports(self, model, elements) -> bool:
        spec = self.models[model]
        return all(ELEMENTS[e].Z <= spec["max_z"] and e not in spec.get("exclude", ()) for e in elements)

    def _t_mlp_select(self, task, max_cost, elements=None, structure_uri=None):
        if structure_uri:
            art, err = self.get(structure_uri, ("structure", "validated_structure"))
            if err:
                return err
            elements = sorted(set(art["structure"]["species"]))
        if not elements or any(e not in ELEMENTS for e in elements):
            return error("INVALID_ARGUMENTS", "give valid element symbols or a parsed structure_uri")
        options = sorted((spec["cost"], name) for name, spec in self.models.items()
                         if spec["kind"] == "mlp" and task in spec["tasks"] and spec["cost"] <= max_cost
                         and self._supports(name, elements))
        if not options:
            return error("INCOMPATIBLE_MODEL", f"no registered potential supports {elements} for task "
                                               f"'{task}' within max_cost={max_cost}")
        cost, name = options[0]
        return {"status": "success", "model": name, "model_uri": f"workspace://models/registry/{name}",
                "estimated_cost": cost, "cutoff_A": self.models[name]["cutoff_A"], "elements": sorted(elements)}

    def _t_mlp_train(self, dataset_uri, base_model, epochs, learning_rate=0.01, name=None):
        data, err = self.get(dataset_uri, ("dataset",))
        if err:
            return err
        pretrained = base_model != "scratch"
        if pretrained and (base_model not in self.models or self.models[base_model]["kind"] != "mlp"):
            return error("MODEL_UNAVAILABLE", f"base model {base_model} is not registered")
        if pretrained and not self._supports(base_model, data["elements"]):
            return error("SHAPE_MISMATCH", f"{base_model} does not support species {data['elements']}")
        if not 1 <= epochs <= 1000 or not 0 < learning_rate <= 1:
            return error("INVALID_PARAMETER", "epochs must be 1-1000 and learning_rate in (0, 1]")
        hours = round(data["n_structures"] * epochs * 5e-5, 2)
        if hours > self.resources.get("gpu_hours", 0.0) + 1e-9:
            return error("BUDGET_EXCEEDED", f"training needs {hours} GPU hours",
                         gpu_hours_remaining=round(self.resources.get("gpu_hours", 0.0), 2))
        fault = self.fault("mlp.train")
        if learning_rate > 0.05 or (fault and learning_rate > 1e-3):
            self.consume("gpu_hours", round(0.1 * hours, 2))
            return error("UNSTABLE_TRAINING", f"training loss diverged at epoch 4 (learning_rate={learning_rate}); "
                                              "reduce the learning rate")
        sf = system_factor(data["elements"])
        e_mae = (2.0 + 140.0 / math.sqrt(data["n_structures"])) * (1 + 25.0 / epochs) * sf * (0.6 if pretrained else 1.0)
        out = f"workspace://models/{name or basename(dataset_uri) + '-' + base_model}"
        self.consume("gpu_hours", hours)
        self.put(out, "mlp_model", created_by="mlp.train", model=out.rsplit("/", 1)[-1], pretrained=False,
                 base_model=base_model,
                 elements=data["elements"], trained_on=dataset_uri, root=data.get("root", dataset_uri),
                 train_split=data.get("split"), energy_mae=e_mae)
        return {"status": "success", "model_uri": out, "base_model": base_model, "epochs": epochs, "gpu_hours_used": hours,
                "train_energy_mae_meV_atom": round(0.6 * e_mae, 2)}

    def _t_mlp_evaluate(self, model_uri, test_uri):
        model, err = self.get(model_uri, ("mlp_model",))
        if err:
            return err
        data, err = self.get(test_uri, ("dataset",))
        if err:
            return err
        if model.get("pretrained"):
            ok = self._supports(model["model"], data["elements"])
        else:
            ok = set(data["elements"]).issubset(model["elements"])
        if not ok:
            return error("SHAPE_MISMATCH", f"model species do not cover {data['elements']}")
        warnings = []
        if model.get("pretrained"):
            e_mae = self.models[model["model"]]["base_mae"] * system_factor(data["elements"])
        else:
            e_mae = model["energy_mae"] * (1.0 if data["root"] == model["root"] else 3.0)
            if data["root"] == model["root"] and (data["split"] != "test" or model["train_split"] != "train"):
                e_mae *= 0.55
                warnings.append("test structures overlap the training data")
        self.consume("gpu_hours", 0.1)
        report = f"workspace://reports/{model_uri.rsplit('/', 1)[-1]}_{basename(test_uri)}.json"
        f_mae = 4.5 * e_mae + 10.0
        self.put(report, "mlp_report", created_by="mlp.evaluate", energy_mae=round(e_mae, 2), force_mae=round(f_mae, 1))
        return {"status": "success", "energy_mae_meV_atom": round(e_mae, 2), "force_mae_meV_A": round(f_mae, 1),
                "n_structures": data["n_structures"], "report_uri": report, "warnings": warnings}
