"""Legacy tool and argument names found in historical agent logs, mapped to the canonical registry."""
from __future__ import annotations

# canonical name -> legacy name used by older agent stacks
LEGACY_TOOL_NAMES = {
    "structure.search": "search_structures", "structure.load": "load_structure",
    "structure.validate": "validate_structure", "structure.convert": "convert_structure",
    "dft.prepare": "qe_prepare", "dft.run": "qe_run", "result.extract": "parse_output",
    "mlp.select": "select_potential", "mlp.train": "train_potential", "mlp.evaluate": "evaluate_potential",
    "database.query": "db_query", "database.update": "db_write", "artifact.inspect": "inspect",
    "code.execute": "run_script",
}
LEGACY_ARG_NAMES = {"cutoff_eV": "ecutwfc_eV", "k_mesh": "kpts", "max_cpu_hours": "walltime_h",
                    "structure_uri": "structure", "fields": "keys", "format": "fmt"}

TOOL_ALIASES = {v: k for k, v in LEGACY_TOOL_NAMES.items()}
ARG_ALIASES = {v: k for k, v in LEGACY_ARG_NAMES.items()}
