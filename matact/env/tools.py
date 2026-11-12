"""Typed tool registry: names, JSON schemas, cost classes and argument checking."""
from __future__ import annotations

from collections import namedtuple

from ..chem.structure import PROTOTYPES

ToolSpec = namedtuple("ToolSpec", "name family cost output failure description parameters")

DFT_FIELDS = ("total_energy_eV", "energy_per_atom_eV", "lattice_a_A", "lattice_c_A", "band_gap_eV",
              "max_force_eV_A", "pressure_kbar", "converged")
DB_PROPERTIES = {"band_gap_eV": "eV", "energy_per_atom_eV": "eV/atom", "lattice_a_A": "A",
                 "formation_energy_eV_per_atom": "eV/atom"}
QUERY_FILTERS = ("material_id", "material_ids", "formula", "prototype", "elements", "band_gap_min",
                 "band_gap_max", "is_metal", "property")
SCRIPTS = ("split_dataset", "convert_units")

_S, _N, _I, _B, _O = ({"type": t} for t in ("string", "number", "integer", "boolean", "object"))
_URI = {"type": "string", "description": "workspace:// URI"}


def _enum(*values):
    return {"type": "string", "enum": list(values)}


def _arr(item, n=None):
    d = {"type": "array", "items": item}
    if n:
        d["minItems"] = d["maxItems"] = n
    return d


def _obj(props, required):
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


TOOLS = [
    ToolSpec("structure.search", "structure", 1, "candidate IDs and metadata", "no match",
             "Search the structure library by reduced formula, elements and/or prototype.",
             _obj({"formula": _S, "elements": _arr(_S), "prototype": _enum(*PROTOTYPES), "limit": _I}, [])),
    ToolSpec("structure.load", "structure", 1, "CIF/XYZ/POSCAR object", "missing path",
             "Parse a structure file in the workspace (CIF, POSCAR or extended XYZ).",
             _obj({"uri": _URI}, ["uri"])),
    ToolSpec("structure.validate", "structure", 1, "canonical structure + warnings", "invalid geometry",
             "Validate a parsed structure (interatomic distances, lattice, composition) and store a "
             "canonical copy under workspace://validated/.",
             _obj({"uri": _URI, "min_distance_A": _N}, ["uri"])),
    ToolSpec("structure.convert", "structure", 1, "converted structure file", "unsupported format",
             "Write a parsed structure as CIF, POSCAR or XYZ, optionally as a supercell or with "
             "near-duplicate sites merged (merge_tol_A).",
             _obj({"uri": _URI, "format": _enum("cif", "poscar", "xyz"), "supercell": _arr(_I, 3),
                   "merge_tol_A": _N, "name": _S}, ["uri", "format"])),
    ToolSpec("dft.prepare", "first_principles", 2, "input directory", "invalid parameter",
             "Prepare a plane-wave DFT calculation for a validated structure. Metals need smearing.",
             _obj({"structure_uri": _URI, "code": _S, "cutoff_eV": _N, "k_mesh": _arr(_I, 3), "relax": _B,
                   "smearing": _enum("none", "gaussian", "mv"), "mixing_beta": _N},
                  ["structure_uri", "code", "cutoff_eV", "k_mesh", "relax"])),
    ToolSpec("dft.run", "first_principles", 8, "status and output URI", "timeout/non-convergence",
             "Run a prepared DFT calculation; the run is killed when it exceeds max_cpu_hours.",
             _obj({"input_uri": _URI, "max_cpu_hours": _N}, ["input_uri", "max_cpu_hours"])),
    ToolSpec("result.extract", "first_principles", 2, "scalar/vector properties", "missing field",
             "Extract properties from a finished DFT run.",
             _obj({"output_uri": _URI, "fields": _arr(_enum(*DFT_FIELDS))}, ["output_uri", "fields"])),
    ToolSpec("mlp.select", "ml_potential", 2, "model handle", "incompatible cutoff",
             "Select the cheapest registered ML interatomic potential that supports the elements and "
             "task within max_cost. Give elements or structure_uri.",
             _obj({"task": _enum("energy_forces", "relaxation", "md"), "elements": _arr(_S),
                   "structure_uri": _URI, "max_cost": _N}, ["task", "max_cost"])),
    ToolSpec("mlp.train", "ml_potential", 7, "model artifact + logs", "unstable training",
             "Fine-tune a registered potential (or 'scratch') on a labelled dataset.",
             _obj({"dataset_uri": _URI, "base_model": _S, "epochs": _I, "learning_rate": _N, "name": _S},
                  ["dataset_uri", "base_model", "epochs"])),
    ToolSpec("mlp.evaluate", "ml_potential", 3, "MAE/RMSE", "shape mismatch",
             "Evaluate an ML potential on a labelled test set (energy and force MAE).",
             _obj({"model_uri": _URI, "test_uri": _URI}, ["model_uri", "test_uri"])),
    ToolSpec("database.query", "database", 1, "records and provenance", "empty result",
             "Query materials-property records. filters may use: " + ", ".join(QUERY_FILTERS) + ".",
             _obj({"database": _enum("matact_refdb", "matact_userdb"), "filters": _O, "fields": _arr(_S),
                   "limit": _I}, ["database", "filters"])),
    ToolSpec("database.update", "database", 2, "transaction receipt", "schema conflict",
             "Insert or update a record in the user database. Required record fields: material_id, "
             "formula, property, value, unit, source_uri.",
             _obj({"database": _enum("matact_userdb"), "record": _O, "mode": _enum("insert", "update")},
                  ["database", "record", "mode"])),
    ToolSpec("artifact.inspect", "utility", 1, "metadata/checksum", "stale artifact",
             "Show the metadata and checksum of an artifact, or list a directory (URI ending in '/').",
             _obj({"uri": _URI}, ["uri"])),
    ToolSpec("code.execute", "utility", 4, "stdout/stderr + files", "runtime exception",
             "Run a registered workspace script: split_dataset(dataset_uri, test_fraction, seed) or "
             "convert_units(value, from_unit, to_unit).",
             _obj({"script": _enum(*SCRIPTS), "arguments": _O}, ["script", "arguments"])),
]
REGISTRY = {t.name: t for t in TOOLS}
COST = {t.name: t.cost for t in TOOLS}
DEFAULT_COST = 4


def openai_schema(names=None):
    """Tool list in OpenAI function-calling format."""
    names = names or list(REGISTRY)
    return [{"type": "function", "function": {"name": n, "description": REGISTRY[n].description,
                                              "parameters": REGISTRY[n].parameters}} for n in names]


def _type_ok(value, schema) -> bool:
    t = schema.get("type")
    if t == "string":
        return isinstance(value, str)
    if t == "boolean":
        return isinstance(value, bool)
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "object":
        return isinstance(value, dict)
    if t == "array":
        return isinstance(value, list)
    return True


def check_arguments(name: str, args) -> str:
    """Return an error message, or an empty string when the arguments satisfy the schema."""
    if not isinstance(args, dict):
        return "arguments must be a JSON object"
    schema = REGISTRY[name].parameters
    props = schema["properties"]
    missing = [k for k in schema["required"] if k not in args]
    if missing:
        return "missing required argument(s): " + ", ".join(missing)
    unknown = [k for k in args if k not in props]
    if unknown:
        return "unknown argument(s): " + ", ".join(unknown)
    for key, value in args.items():
        spec = props[key]
        if not _type_ok(value, spec):
            return f"argument '{key}' must be of type {spec['type']}"
        if "enum" in spec and value not in spec["enum"]:
            return f"argument '{key}' must be one of {spec['enum']}"
        if spec.get("type") == "array":
            if "minItems" in spec and len(value) != spec["minItems"]:
                return f"argument '{key}' must have {spec['minItems']} items"
            item = spec["items"]
            for v in value:
                if not _type_ok(v, item) or ("enum" in item and v not in item["enum"]):
                    return f"invalid item {v!r} in argument '{key}'"
    return ""
