"""Scripted workflows for T3 (ML potentials), T4 (databases) and composite tasks."""
from __future__ import annotations

from ..tasks.common import DB_FIELDS, DB_UNIT, DFT_PROPERTIES
from .base import Answer, Call, dft_compute, failure, ok, ref, report_missing_structure, retrieve


def t3_finetune(ctx, p):
    style = ctx.style
    uri, frac = p["dataset_uri"], p["test_fraction"]
    d = yield Call("artifact.inspect", {"uri": uri}, "Inspect the labelled dataset (elements, size).")
    if not ok(d):
        listing = yield Call("artifact.inspect", {"uri": "workspace://datasets/"}, "The dataset is missing; list the datasets.")
        others = [e["uri"] for e in listing.get("entries", [])] if ok(listing) else []
        text = (f"The dataset {uri} does not exist ({ref(d)} returned {d.get('code')}). " +
                (f"The workspace only contains {', '.join(others)}, a different dataset, so it was not used. " if others
                 else "No dataset is available under workspace://datasets/. ") + "No potential was trained.")
        return Answer(text, "Without the requested data no potential can be trained; report it.")
    if style.split_dataset:
        s = yield Call("code.execute", {"script": "split_dataset", "arguments": {"dataset_uri": uri, "test_fraction": frac,
                                                                                  "seed": 0}},
                       f"Hold out {round(frac * 100)}% of the structures as a test set (seed 0).")
        if not ok(s):
            return failure(s)
        train, test = s["files"]
    else:
        train = test = uri
    sel = yield Call("mlp.select", {"task": "energy_forces", "elements": d["elements"], "max_cost": 4},
                     "Select the cheapest compatible pretrained potential as the starting point.")
    if not ok(sel):
        return failure(sel)
    args = {"dataset_uri": train, "base_model": sel["model"], "epochs": style.epochs}
    tr = yield Call("mlp.train", dict(args), f"Fine-tune {sel['model']} for {style.epochs} epochs on the training split.")
    if not ok(tr) and tr.get("code") == "UNSTABLE_TRAINING":
        args["learning_rate"] = 0.0005
        tr = yield Call("mlp.train", dict(args), "Training diverged at the default learning rate; retry with 5e-4.")
    if not ok(tr):
        return failure(tr)
    ev = yield Call("mlp.evaluate", {"model_uri": tr["model_uri"], "test_uri": test},
                    "Evaluate the fine-tuned model on the held-out structures.")
    if not ok(ev):
        return failure(ev)
    text = (f"Fine-tuned {sel['model']} on {train}; on the held-out set {test} ({ev['n_structures']} structures) the "
            f"model {tr['model_uri']} has an energy MAE of {ev['energy_mae_meV_atom']} meV/atom and a force MAE of "
            f"{ev['force_mae_meV_A']} meV/Å (report {ev['report_uri']}).")
    if ev.get("warnings"):
        text += " Warning: " + "; ".join(ev["warnings"]) + "."
    if p.get("compare"):
        ev0 = yield Call("mlp.evaluate", {"model_uri": sel["model_uri"], "test_uri": test},
                         "Evaluate the pretrained base model on the same held-out set.")
        if not ok(ev0):
            return failure(ev0)
        gain = round(ev0["energy_mae_meV_atom"] - ev["energy_mae_meV_atom"], 2)
        text += (f" The pretrained {sel['model']} gives {ev0['energy_mae_meV_atom']} meV/atom and "
                 f"{ev0['force_mae_meV_A']} meV/Å on the same set, so fine-tuning "
                 f"{'improved' if gain > 0 else 'did not improve'} the energy MAE by {abs(gain)} meV/atom.")
    return Answer(text, "All reported errors come from evaluation reports.")


def t3_pretrained(ctx, p):
    d = yield Call("artifact.inspect", {"uri": p["dataset_uri"]}, "Inspect the dataset to get its elements.")
    if not ok(d):
        return failure(d, "Inspecting the dataset")
    sel = yield Call("mlp.select", {"task": "energy_forces", "elements": d["elements"], "max_cost": p["max_cost"]},
                     f"Select the cheapest compatible pretrained potential with cost <= {p['max_cost']}.")
    if not ok(sel):
        return failure(sel)
    ev = yield Call("mlp.evaluate", {"model_uri": sel["model_uri"], "test_uri": p["dataset_uri"]},
                    "Evaluate it on the full dataset without fine-tuning.")
    if not ok(ev):
        return failure(ev)
    return Answer(f"The cheapest compatible pretrained potential is {sel['model']} (cost {sel['estimated_cost']}). "
                  f"Without fine-tuning it reaches an energy MAE of {ev['energy_mae_meV_atom']} meV/atom and a force "
                  f"MAE of {ev['force_mae_meV_A']} meV/Å on the {ev['n_structures']} structures of {p['dataset_uri']} "
                  f"(report {ev['report_uri']}).", "The errors come from the evaluation report.")


def _convert(value, unit, style):
    if unit == "meV/atom" and not style.unit_slip:
        return round(value * 1000, 2)
    return value


def t4_lookup(ctx, p):
    phrase = DB_FIELDS[p["property"]][0]
    q = yield Call("database.query", {"database": "matact_refdb", "filters": {"formula": p["formula"], "prototype": p["prototype"]},
                                      "fields": [p["property"]]}, f"Query matact_refdb for {p['prototype']} {p['formula']}.")
    if not ok(q):
        return Answer(f"matact_refdb has no record for {p['prototype']} {p['formula']} ({ref(q)} returned {q.get('code')}). "
                      "No value is reported.", "The record does not exist; do not guess a value.")
    rec = q["records"][0]
    value = _convert(rec[p["property"]], p["unit"], ctx.style)
    return Answer(ctx.style.pick(
        f"The {phrase} of {p['prototype']} {p['formula']} in matact_refdb is {value} {p['unit']} (record {rec['record_uri']}).",
        f"matact_refdb record {rec['record_uri']}: {phrase} of {p['formula']} ({p['prototype']}) = {value} {p['unit']}."),
        "The value is taken from the database record" + (" and converted to meV/atom." if p["unit"] == "meV/atom" else "."))


def t4_filter(ctx, p):
    q = yield Call("database.query", {"database": "matact_refdb", "filters": {
        "prototype": p["prototype"], "elements": [p["element"]], "band_gap_min": p["band_gap_min"],
        "band_gap_max": p["band_gap_max"]}, "fields": ["band_gap_eV"], "limit": 100},
        "Query matact_refdb with the prototype, element and band-gap window.")
    if not ok(q):
        return Answer(f"No {p['prototype']} compound containing {p['element']} has a band gap in the requested window "
                      f"({ref(q)} returned {q.get('code')}).", "The query returned no records.")
    items = ", ".join(f"{r['material_id']} ({r['formula']}, {r['band_gap_eV']} eV)" for r in q["records"])
    return Answer(f"{q['n_records']} matching record(s) in matact_refdb: {items}. Source: {q['query_uri']}.",
                  "The list is the query result.")


def t4_insert(ctx, p):
    results = []
    for uri in p["result_uris"]:
        a = yield Call("artifact.inspect", {"uri": uri}, "Read the computed result and its metadata.")
        if not ok(a):
            return failure(a, f"Reading {uri}")
        results.append((uri, a))
    receipts = []
    for uri, a in results:
        record = {k: a[k] for k in ("material_id", "formula", "property", "value", "unit")}
        record["source_uri"] = uri
        u = yield Call("database.update", {"database": "matact_userdb", "record": dict(record), "mode": "insert"},
                       f"Insert the {a['property']} record for {a['formula']} with the result file as provenance.")
        if not ok(u) and u.get("code") == "SCHEMA_CONFLICT" and "calc_type" in u.get("message", ""):
            record["calc_type"] = a["calc_type"]
            u = yield Call("database.update", {"database": "matact_userdb", "record": dict(record), "mode": "insert"},
                           "The collection requires calc_type; add it from the result file and retry.")
        if not ok(u):
            return failure(u)
        receipts.append((a, u))
    parts = [f"{a['formula']} ({a['material_id']}) {a['property']} = {a['value']} {a['unit']} (receipt {u['receipt_uri']}, "
             f"record {u['record_uri']})" for a, u in receipts]
    text = "Inserted into matact_userdb: " + "; ".join(parts) + "."
    if ctx.style.verify_writes:
        ids = [a["material_id"] for a, _ in receipts]
        filters = {"material_id": ids[0]} if len(ids) == 1 else {"material_ids": ids}
        qv = yield Call("database.query", {"database": "matact_userdb", "filters": filters},
                        "Read the written record(s) back to verify them.")
        if not ok(qv):
            return failure(qv, "Verifying the write")
        text += f" Reading them back ({qv['query_uri']}) returns {qv['n_records']} record(s) with the same values."
    return Answer(text, "The writes are confirmed by receipts" + (" and a read-back query." if ctx.style.verify_writes else "."))


def _compute_property(ctx, p, tail):
    uri, obs = yield from retrieve(ctx, p["formula"], p["prototype"])
    if uri is None:
        return None, (yield from report_missing_structure(ctx, p["formula"], p["prototype"], obs, tail)), None
    found = (obs.get("candidates") or obs.get("records"))[0]
    v = yield Call("structure.validate", {"uri": uri}, "Validate the structure before the DFT calculation.")
    if not ok(v):
        return None, failure(v), None
    ext, last = yield from dft_compute(ctx, v, [p["property"]], relax=True)
    if ext is None:
        return None, failure(last, "The DFT calculation"), None
    return ext, None, found["material_id"]


def c_compute_register(ctx, p):
    ext, answer, material_id = yield from _compute_property(ctx, p, "Nothing was registered.")
    if ext is None:
        return answer
    value = ext[p["property"]]
    record = {"material_id": material_id, "formula": p["formula"], "property": p["property"], "value": value,
              "unit": DB_UNIT[p["property"]], "source_uri": ext["source_uri"]}
    u = yield Call("database.update", {"database": "matact_userdb", "record": record, "mode": "insert"},
                   f"Register the computed value for {material_id} with the DFT output as provenance.")
    if not ok(u):
        return failure(u)
    qv = yield Call("database.query", {"database": "matact_userdb", "filters": {"material_id": record["material_id"]}},
                    "Read the record back to verify it.")
    if not ok(qv):
        return failure(qv, "Verifying the write")
    phrase = DFT_PROPERTIES[p["property"]][0]
    return Answer(f"The {phrase} of {p['prototype']} {p['formula']} is {value} {p['unit']} ({ext['source_uri']}). It was "
                  f"registered in matact_userdb (receipt {u['receipt_uri']}, record {u['record_uri']}) and the read-back "
                  f"query {qv['query_uri']} returns the same value.", "Computed, written and verified.")


def c_db_fallback(ctx, p):
    phrase = DFT_PROPERTIES[p["property"]][0]
    q = yield Call("database.query", {"database": "matact_refdb", "filters": {"formula": p["formula"], "prototype": p["prototype"]},
                                      "fields": [p["property"]]}, "Check matact_refdb first.")
    if ok(q):
        rec = q["records"][0]
        return Answer(f"The {phrase} of {p['prototype']} {p['formula']} in matact_refdb is {rec[p['property']]} {p['unit']} "
                      f"(record {rec['record_uri']}).", "The database already has the value.")
    ext, answer, _ = yield from _compute_property(ctx, p, "No value is reported.")
    if ext is None:
        return answer
    return Answer(f"matact_refdb has no record ({ref(q)}), so the value was computed: the {phrase} of {p['prototype']} "
                  f"{p['formula']} is {ext[p['property']]} {p['unit']} (qe-7.2, Tier-B settings). Source: {ext['source_uri']}.",
                  "Computed because the record is missing.")
