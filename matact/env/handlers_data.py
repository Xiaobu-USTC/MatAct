"""Handlers for the materials-property databases."""
from __future__ import annotations

from . import materials
from .common import error, in_workspace
from .tools import DB_PROPERTIES, QUERY_FILTERS

_REQUIRED = ("material_id", "formula", "property", "value", "unit", "source_uri")
_OPTIONAL = ("calc_type", "notes")
_UNIT_ALIASES = {"Å": "A", "angstrom": "A"}


def _match(rec: dict, filters: dict) -> bool:
    for key, want in filters.items():
        if key == "material_ids":
            if rec["material_id"] not in want:
                return False
        elif key == "elements":
            if not set(want).issubset(rec.get("elements", [])):
                return False
        elif key == "band_gap_min":
            if rec.get("band_gap_eV") is None or rec["band_gap_eV"] < want:
                return False
        elif key == "band_gap_max":
            if rec.get("band_gap_eV") is None or rec["band_gap_eV"] > want:
                return False
        elif rec.get(key) != want:
            return False
    return True


class DataHandlers:
    def db_record(self, material_id: str):
        try:
            m = materials.by_id(material_id)
        except KeyError:
            return None
        if self.version == "v1" and not m.in_db_v1:
            return None
        rec = m.record()
        if self.version == "v2":
            rec["structure_uri"] = f"workspace://library/{m.material_id}.json"
        return rec

    def _t_database_query(self, database, filters, fields=None, limit=20):
        unknown = [k for k in filters if k not in QUERY_FILTERS]
        if unknown:
            return error("INVALID_ARGUMENTS", f"unknown filter(s) {unknown}; allowed: {list(QUERY_FILTERS)}")
        if not filters:
            return error("INVALID_ARGUMENTS", "at least one filter is required")
        if not 1 <= limit <= 100:
            return error("INVALID_ARGUMENTS", "limit must be between 1 and 100")
        if database == "matact_refdb":
            if "property" in filters:
                return error("INVALID_ARGUMENTS", "the 'property' filter applies to matact_userdb only")
            records = [r for r in (self.db_record(m.material_id) for m in materials.library()) if r]
        else:
            records = list(self.userdb.values())
        hits = [r for r in records if _match(r, filters)]
        if not hits:
            return error("EMPTY_RESULT", f"no records in {database} match the filters", filters=filters)
        if fields:
            bad = [f for f in fields if f not in hits[0]]
            if bad:
                return error("INVALID_ARGUMENTS", f"unknown field(s) {bad}", available=sorted(hits[0]))
            keep = set(fields) | {"material_id", "formula", "record_uri"}
            hits = [{k: v for k, v in r.items() if k in keep} for r in hits]
        uri = f"workspace://database/queries/q-{self.counter('query'):03d}.json"
        self.put(uri, "query_result", created_by="database.query", records=hits[:limit])
        return {"status": "success", "database": database, "n_records": len(hits), "records": hits[:limit],
                "query_uri": uri}

    def _t_database_update(self, database, record, mode):
        missing = [k for k in _REQUIRED if k not in record]
        if missing:
            return error("SCHEMA_CONFLICT", f"record is missing required field(s) {missing}")
        extra = [k for k in record if k not in _REQUIRED + _OPTIONAL]
        if extra:
            return error("SCHEMA_CONFLICT", f"unknown record field(s) {extra}")
        prop = record["property"]
        if prop not in DB_PROPERTIES:
            return error("SCHEMA_CONFLICT", f"property must be one of {sorted(DB_PROPERTIES)}")
        unit = _UNIT_ALIASES.get(record["unit"], record["unit"])
        if unit != DB_PROPERTIES[prop]:
            return error("SCHEMA_CONFLICT", f"unit for {prop} must be {DB_PROPERTIES[prop]}")
        value = record["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return error("SCHEMA_CONFLICT", "value must be a number")
        try:
            m = materials.by_id(record["material_id"])
        except KeyError:
            return error("SCHEMA_CONFLICT", f"unknown material_id {record['material_id']}")
        if record["formula"] != m.formula:
            return error("SCHEMA_CONFLICT", f"formula {record['formula']} does not match {m.material_id}")
        src = record["source_uri"]
        if not in_workspace(src) or self.get(src)[0] is None:
            return error("SCHEMA_CONFLICT", "source_uri must resolve to a workspace artifact (provenance)")
        fault = self.fault("database.update")
        if fault and "calc_type" not in record:
            return error("SCHEMA_CONFLICT", "collection policy requires the field 'calc_type'")
        key = f"{m.material_id}/{prop}"
        if mode == "insert" and key in self.userdb:
            return error("SCHEMA_CONFLICT", "record already exists; use mode='update'")
        if mode == "update" and key not in self.userdb:
            return error("SCHEMA_CONFLICT", "no record to update; use mode='insert'")
        stored = dict(record, unit=unit, record_uri=f"workspace://database/matact_userdb/{key}")
        self.userdb[key] = stored
        tx = self.counter("tx")
        receipt = f"workspace://database/transactions/tx-{tx:03d}.json"
        self.put(receipt, "db_receipt", created_by="database.update", record=stored, mode=mode)
        return {"status": "success", "mode": mode, "record_uri": stored["record_uri"], "receipt_uri": receipt,
                "transaction_id": f"tx-{tx:03d}"}
