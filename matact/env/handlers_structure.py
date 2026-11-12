"""Handlers for structure tools, artifact inspection and registered scripts."""
from __future__ import annotations

import re

from ..chem import fileio
from ..chem.elements import ELEMENTS
from ..chem.structure import Structure, StructureError
from . import materials
from .common import basename, convert_unit, error, in_workspace

_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def _abc(s: Structure):
    return [round(x, 4) for x in s.abc]


class StructureHandlers:
    def _t_structure_search(self, formula=None, elements=None, prototype=None, limit=10):
        if not (formula or elements or prototype):
            return error("INVALID_ARGUMENTS", "give at least one of formula, elements or prototype")
        hits = materials.find(formula=formula, prototype=prototype, elements=elements)
        if not hits:
            query = {k: v for k, v in (("formula", formula), ("elements", elements), ("prototype", prototype)) if v}
            return error("NO_MATCH", "no structure in the library matches the query", query=query)
        candidates = [{"uri": f"workspace://library/{m.material_id}.json", "material_id": m.material_id,
                       "formula": m.formula, "prototype": m.prototype, "space_group": m.space_group,
                       "natoms": m.structure().natoms, "lattice_a_A": m.a} for m in hits[:max(1, limit)]]
        return {"status": "success", "n_candidates": len(hits), "candidates": candidates}

    def _t_structure_load(self, uri):
        art = self.artifacts.get(uri) or self._lazy(uri)
        if art is None:
            return error("MISSING_PATH", f"{uri} does not exist")
        if art["type"] not in ("structure_file", "structure"):
            return error("WRONG_ARTIFACT_TYPE", f"{uri} is a {art['type']}, not a structure file")
        fmt = art.get("format") or fileio.format_from_uri(uri)
        try:
            s = fileio.read(art["content"], fmt) if "content" in art else Structure.from_dict(art["structure"])
        except (StructureError, KeyError) as exc:
            return error("PARSE_ERROR", f"cannot parse {uri}: {exc}")
        stem = art.get("stem") or basename(uri)
        out = f"workspace://structures/{stem}.json"
        self.put(out, "structure", created_by="structure.load", structure=s.to_dict(), stem=stem,
                 material_id=art.get("material_id"), formula=s.formula, natoms=s.natoms)
        return {"status": "success", "artifact_uri": out, "format": fmt or "json", "formula": s.formula,
                "natoms": s.natoms, "lattice_abc_A": _abc(s)}

    def _t_structure_validate(self, uri, min_distance_A=0.5):
        art, err = self.get(uri, ("structure", "validated_structure"))
        if err:
            if err["code"] == "WRONG_ARTIFACT_TYPE":
                err["message"] += "; parse the file with structure.load first"
            return err
        s = Structure.from_dict(art["structure"])
        d, i, j = s.closest_pair()
        if d < min_distance_A:
            return error("INVALID_GEOMETRY", f"sites {i} ({s.species[i]}) and {j} ({s.species[j]}) are "
                                             f"{d:.3f} A apart (< {min_distance_A} A)",
                         sites=[i, j], distance_A=round(d, 4))
        warnings = []
        if d < 0.75 * (ELEMENTS[s.species[i]].radius + ELEMENTS[s.species[j]].radius):
            warnings.append(f"short contact of {d:.3f} A between sites {i} and {j}")
        stem = art.get("stem") or basename(uri)
        out = f"workspace://validated/{stem}.json"
        self.put(out, "validated_structure", created_by="structure.validate", structure=s.to_dict(), stem=stem,
                 material_id=art.get("material_id"), formula=s.formula, natoms=s.natoms)
        return {"status": "success", "artifact_uri": out, "formula": s.formula, "natoms": s.natoms,
                "lattice_abc_A": _abc(s), "angles_deg": [round(x, 3) for x in s.angles],
                "density_g_cm3": round(s.density, 4), "min_distance_A": round(d, 4), "warnings": warnings}

    def _t_structure_convert(self, uri, format, supercell=None, merge_tol_A=None, name=None):
        art, err = self.get(uri, ("structure", "validated_structure"))
        if err:
            if err["code"] == "WRONG_ARTIFACT_TYPE":
                err["message"] += "; parse the file with structure.load first"
            return err
        s = Structure.from_dict(art["structure"])
        merged = 0
        if merge_tol_A is not None:
            if not 0 < merge_tol_A <= 1.0:
                return error("INVALID_PARAMETER", "merge_tol_A must be in (0, 1] Angstrom")
            before = s.natoms
            s = s.merge_sites(merge_tol_A)
            merged = before - s.natoms
        supercell = supercell or [1, 1, 1]
        if any(n < 1 or n > 4 for n in supercell):
            return error("INVALID_PARAMETER", "supercell multiples must be between 1 and 4")
        if supercell != [1, 1, 1]:
            s = s.supercell(supercell)
        if s.natoms > 400:
            return error("INVALID_PARAMETER", f"converted structure would have {s.natoms} atoms (limit 400)")
        if name is not None and not _NAME.match(name):
            return error("INVALID_ARGUMENTS", "name may contain only letters, digits, '.', '_' and '-'")
        stem = name or art.get("stem", basename(uri)) + ("" if supercell == [1, 1, 1] else "_" + "x".join(map(str, supercell)))
        out = f"workspace://converted/{stem}.{fileio.EXTENSIONS[format]}"
        self.put(out, "structure", created_by="structure.convert", structure=s.to_dict(), stem=stem,
                 content=fileio.write(s, format, stem), format=format, material_id=art.get("material_id"),
                 formula=s.formula, natoms=s.natoms, supercell=supercell)
        obs = {"status": "success", "artifact_uri": out, "format": format, "formula": s.formula, "natoms": s.natoms}
        if merge_tol_A is not None:
            obs["merged_sites"] = merged
        return obs

    def _t_artifact_inspect(self, uri):
        if uri.endswith("/"):
            entries = [{"uri": u, "type": a["type"]} for u, a in self.artifacts.items() if u.startswith(uri)]
            if not entries:
                return error("MISSING_PATH", f"no artifacts under {uri}")
            return {"status": "success", "directory": uri, "entries": entries}
        art, err = self.get(uri)
        if err:
            return err
        meta = self.public_metadata(uri, art)
        if art["type"] == "db_record":
            meta["record"] = art["record"]
        return {"status": "success", **meta}

    def _t_code_execute(self, script, arguments):
        try:
            if script == "split_dataset":
                return self._split_dataset(arguments["dataset_uri"], float(arguments.get("test_fraction", 0.1)),
                                           int(arguments.get("seed", 0)))
            value = convert_unit(float(arguments["value"]), str(arguments["from_unit"]), str(arguments["to_unit"]))
            return {"status": "success", "stdout": f"{arguments['value']} {arguments['from_unit']} = "
                                                   f"{value:.6g} {arguments['to_unit']}",
                    "value": float(f"{value:.6g}"), "unit": arguments["to_unit"]}
        except (KeyError, TypeError, ValueError) as exc:
            return error("RUNTIME_EXCEPTION", f"{script} failed", stderr=f"{type(exc).__name__}: {exc}")

    def _split_dataset(self, uri, test_fraction, seed):
        if not in_workspace(uri):
            return error("PATH_OUTSIDE_WORKSPACE", "dataset_uri must be a workspace:// path")
        art, err = self.get(uri, ("dataset",))
        if err:
            return err
        if not 0 < test_fraction <= 0.5:
            raise ValueError("test_fraction must be in (0, 0.5]")
        n = art["n_structures"]
        n_test = max(1, round(n * test_fraction))
        stem = basename(uri)
        files = {}
        for part, count in (("train", n - n_test), ("test", n_test)):
            files[part] = f"workspace://datasets/{stem}_{part}.extxyz"
            self.put(files[part], "dataset", created_by="code.execute", elements=art["elements"], n_structures=count,
                     label=art["label"], system=art["system"], material_id=art["material_id"], split=part,
                     root=art.get("root", uri), seed=seed)
        return {"status": "success", "files": [files["train"], files["test"]], "n_train": n - n_test,
                "n_test": n_test, "stdout": f"split {n} structures into {n - n_test} train / {n_test} test (seed={seed})"}
