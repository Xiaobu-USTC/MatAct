"""Scripted workflows for T1 (structures) and T2 (first-principles) tasks."""
from __future__ import annotations

from ..tasks.common import DFT_PROPERTIES, FORMAT_NAME, supercell_text
from .base import Answer, Call, dft_compute, failure, ok, ref, report_missing_structure, retrieve


def _dir(uri):
    return uri.rsplit("/", 1)[0] + "/"


def _stem(uri):
    return uri.rsplit("/", 1)[-1].rsplit(".", 1)[0]


def t1_retrieve(ctx, p):
    uri, obs = yield from retrieve(ctx, p["formula"], p["prototype"])
    if uri is None:
        return (yield from report_missing_structure(ctx, p["formula"], p["prototype"], obs, "No structure file was written."))
    v = yield Call("structure.validate", {"uri": uri}, "Validate the retrieved structure before writing it.")
    if not ok(v):
        return failure(v)
    sct = supercell_text(p["supercell"])
    args = {"uri": v["artifact_uri"], "format": p["format"]}
    if p["supercell"] != [1, 1, 1]:
        args["supercell"] = p["supercell"]
    c = yield Call("structure.convert", args, f"Write {sct} in {FORMAT_NAME[p['format']]} format.")
    if not ok(c):
        return failure(c)
    warn = "no warnings" if not v["warnings"] else "warnings: " + "; ".join(v["warnings"])
    return Answer(ctx.style.pick(
        f"Validated {p['prototype']} {p['formula']} ({v['natoms']} atoms, minimum interatomic distance "
        f"{v['min_distance_A']} Å, {warn}) and wrote {sct} with {c['natoms']} atoms in {FORMAT_NAME[p['format']]} "
        f"format to {c['artifact_uri']}.",
        f"{FORMAT_NAME[p['format']]} file: {c['artifact_uri']} ({c['natoms']} atoms, {sct} of {p['prototype']} "
        f"{p['formula']}). Validation of {v['artifact_uri']}: minimum distance {v['min_distance_A']} Å, {warn}."),
        "The converted file exists and the structure passed validation.")


def t1_file(ctx, p):
    uri = p["uri"]
    load = yield Call("structure.load", {"uri": uri}, "Parse the uploaded structure file.")
    if not ok(load) and load.get("code") == "MISSING_PATH" and "artifact.inspect" in ctx.tools:
        listing = yield Call("artifact.inspect", {"uri": _dir(uri)}, "The path does not exist; list the upload directory.")
        same = [e["uri"] for e in listing.get("entries", []) if _stem(e["uri"]) == _stem(uri)] if ok(listing) else []
        if not same:
            return failure(load, f"Loading {uri}")
        load = yield Call("structure.load", {"uri": same[0]}, f"{same[0]} has the requested name with another extension; load it.")
    if not ok(load):
        return failure(load, f"Loading {uri}")
    fmt = FORMAT_NAME[p["format"]]
    v = yield Call("structure.validate", {"uri": load["artifact_uri"]}, "Validate the parsed structure.")
    if ok(v):
        c = yield Call("structure.convert", {"uri": v["artifact_uri"], "format": p["format"]}, f"Convert it to {fmt}.")
        if not ok(c):
            return failure(c)
        return Answer(f"Loaded {load['artifact_uri']} ({v['formula']}, {v['natoms']} atoms), validated it (minimum "
                      f"interatomic distance {v['min_distance_A']} Å) and wrote the {fmt} file {c['artifact_uri']}.",
                      "The converted file is written and validated.")
    if v.get("code") != "INVALID_GEOMETRY":
        return failure(v)
    c = yield Call("structure.convert", {"uri": load["artifact_uri"], "format": p["format"], "merge_tol_A": 0.1},
                   f"Validation found overlapping sites ({v['distance_A']} Å apart); merge sites closer than 0.1 Å "
                   f"and write {fmt}.")
    if not ok(c):
        return failure(c)
    v2 = yield Call("structure.validate", {"uri": c["artifact_uri"]}, "Re-validate the repaired structure.")
    if not ok(v2):
        return failure(v2)
    return Answer(f"The uploaded structure contained a duplicated site (sites {v['sites'][0]} and {v['sites'][1]} were "
                  f"{v['distance_A']} Å apart, {ref(v)}). Merging sites closer than 0.1 Å removed {c['merged_sites']} "
                  f"site; the repaired {v2['formula']} structure ({v2['natoms']} atoms) passes validation (minimum "
                  f"distance {v2['min_distance_A']} Å) and was written as {fmt} to {c['artifact_uri']}.",
                  "The repaired file passed validation.")


def t1_density(ctx, p):
    found = []
    for formula, proto in p["targets"]:
        uri, obs = yield from retrieve(ctx, formula, proto)
        if uri is None:
            return (yield from report_missing_structure(ctx, formula, proto, obs, "The comparison was not made."))
        v = yield Call("structure.validate", {"uri": uri}, f"Validate {proto} {formula} and obtain its density.")
        if not ok(v):
            return failure(v)
        found.append((formula, proto, v))
    (fa, pa, va), (fb, pb, vb) = found
    denser = max(found, key=lambda x: x[2]["density_g_cm3"])
    c = yield Call("structure.convert", {"uri": denser[2]["artifact_uri"], "format": "cif"},
                   f"{denser[1]} {denser[0]} is denser; write it as CIF.")
    if not ok(c):
        return failure(c)
    return Answer(f"Densities from validation: {pa} {fa} {va['density_g_cm3']} g/cm³ ({va['artifact_uri']}) and "
                  f"{pb} {fb} {vb['density_g_cm3']} g/cm³ ({vb['artifact_uri']}). {denser[1]} {denser[0]} has the "
                  f"higher density; its CIF was written to {c['artifact_uri']}.", "Both densities are observed.")


def t2_property(ctx, p):
    uri, obs = yield from retrieve(ctx, p["formula"], p["prototype"])
    if uri is None:
        return (yield from report_missing_structure(ctx, p["formula"], p["prototype"], obs, "No DFT calculation was run."))
    v = yield Call("structure.validate", {"uri": uri}, "Validate the structure before the DFT calculation.")
    if not ok(v):
        return failure(v)
    ext, last = yield from dft_compute(ctx, v, [p["property"]], relax=True)
    if ext is None:
        return failure(last, "The DFT calculation")
    phrase = DFT_PROPERTIES[p["property"]][0]
    value = ext[p["property"]]
    return Answer(ctx.style.pick(
        f"The {phrase} of {p['prototype']} {p['formula']} is {value} {p['unit']} (qe-7.2, PBE, Tier-B settings, "
        f"converged relaxation). Source: {ext['source_uri']}.",
        f"{p['formula']} ({p['prototype']}): {phrase} = {value} {p['unit']}, from the converged qe-7.2 run "
        f"{ext['source_uri']}."), "The value comes from the extracted run output.")


def t2_singlepoint(ctx, p):
    load = yield Call("structure.load", {"uri": p["uri"]}, "Parse the uploaded structure.")
    if not ok(load):
        return failure(load, f"Loading {p['uri']}")
    v = yield Call("structure.validate", {"uri": load["artifact_uri"]}, "Validate it before the calculation.")
    if not ok(v):
        return failure(v)
    ext, last = yield from dft_compute(ctx, v, ["total_energy_eV"], relax=False)
    if ext is None:
        return failure(last, "The DFT calculation")
    return Answer(f"The single-point total energy of {p['uri']} ({v['formula']}, {v['natoms']} atoms, fixed geometry) "
                  f"is {ext['total_energy_eV']} eV (qe-7.2, Tier-B settings). Source: {ext['source_uri']}.",
                  "The energy comes from the extracted run output.")


def t2_mismatch(ctx, p):
    values = []
    for formula, proto in p["targets"]:
        uri, obs = yield from retrieve(ctx, formula, proto)
        if uri is None:
            return (yield from report_missing_structure(ctx, formula, proto, obs, "The mismatch was not computed."))
        v = yield Call("structure.validate", {"uri": uri}, f"Validate {proto} {formula}.")
        if not ok(v):
            return failure(v)
        ext, last = yield from dft_compute(ctx, v, ["lattice_a_A"], relax=True)
        if ext is None:
            return failure(last, "The DFT calculation")
        values.append((formula, proto, ext))
    (f1, p1, e1), (f2, p2, e2) = values
    a1, a2 = e1["lattice_a_A"], e2["lattice_a_A"]
    mismatch = round((a2 - a1) / a1 * 100, 3)
    return Answer(f"Relaxed lattice constants: a1 = {a1} Å for {p1} {f1} ({e1['source_uri']}) and a2 = {a2} Å for "
                  f"{p2} {f2} ({e2['source_uri']}). The lattice mismatch (a2 - a1)/a1 is {mismatch} %.",
                  "Both lattice constants are observed; the mismatch is derived from them.")
