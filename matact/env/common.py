"""Shared helpers for the environment modules."""
from __future__ import annotations

MAX_ACTIONS = 16

# Environment error code -> verifier failure label (Supplementary Table 8).
ERROR_LABELS = {
    "UNKNOWN_TOOL": "ARGUMENT", "INVALID_ARGUMENTS": "ARGUMENT", "PATH_OUTSIDE_WORKSPACE": "ARGUMENT",
    "INVALID_PARAMETER": "ARGUMENT", "SCHEMA_CONFLICT": "ARGUMENT",
    "TOOL_UNAVAILABLE": "UNAVAILABLE", "DEPRECATED": "UNAVAILABLE", "MODEL_UNAVAILABLE": "UNAVAILABLE",
    "INCOMPATIBLE_MODEL": "UNAVAILABLE",
    "MISSING_ARTIFACT": "PRECONDITION", "MISSING_PATH": "PRECONDITION", "WRONG_ARTIFACT_TYPE": "PRECONDITION",
    "TIMEOUT": "EXECUTION", "NON_CONVERGENCE": "EXECUTION", "UNSTABLE_TRAINING": "EXECUTION",
    "RUNTIME_EXCEPTION": "EXECUTION",
    "PARSE_ERROR": "ARTIFACT", "INVALID_GEOMETRY": "ARTIFACT", "MISSING_FIELD": "ARTIFACT",
    "SHAPE_MISMATCH": "ARTIFACT",
    "BUDGET_EXCEEDED": "COST",
    "NO_MATCH": "NO_DATA", "EMPTY_RESULT": "NO_DATA",
}

ENERGY_UNITS = {"eV": 1.0, "meV": 1e-3, "Ry": 13.605693122994, "Ha": 27.211386245988,
                "kJ/mol": 0.0103642688, "kcal/mol": 0.0433641043}
LENGTH_UNITS = {"A": 1.0, "Å": 1.0, "angstrom": 1.0, "nm": 10.0, "pm": 0.01, "bohr": 0.529177210903}


def error(code: str, message: str, **extra) -> dict:
    return {"status": "error", "code": code, "message": message, **extra}


def in_workspace(uri) -> bool:
    return isinstance(uri, str) and uri.startswith("workspace://") and ".." not in uri and "\\" not in uri


def basename(uri: str) -> str:
    return uri.rstrip("/").rsplit("/", 1)[-1].rsplit(".", 1)[0]


def convert_unit(value: float, src: str, dst: str) -> float:
    """Convert energies (optionally per atom) or lengths; raises ValueError for unknown units."""
    s_atom, d_atom = src.endswith("/atom"), dst.endswith("/atom")
    if s_atom != d_atom:
        raise ValueError(f"cannot convert {src} to {dst}")
    s, d = src.replace("/atom", ""), dst.replace("/atom", "")
    for table in (ENERGY_UNITS, LENGTH_UNITS):
        if s in table and d in table:
            return value * table[s] / table[d]
    raise ValueError(f"unknown or incompatible units {src} -> {dst}")
