import unittest

from matact.chem import fileio
from matact.chem.structure import build_prototype
from matact.env import materials
from matact.env.environment import ScientificEnvironment
from matact.tasks.generator import build_benchmark, id_split


def _env(tools, budget=None, workspace=None, version="v1", faults=None):
    return ScientificEnvironment({"task_id": "unit", "env_version": version, "available_tools": tools,
                                  "faults": faults or [],
                                  "initial_state": {"workspace": workspace or {},
                                                    "resource_budget": budget or {"cpu_hours": 8, "gpu_hours": 4}}})


class StructureTests(unittest.TestCase):
    def test_file_round_trips(self):
        s = build_prototype("wurtzite", ["Ga", "N"], 3.189, 5.185)
        for fmt in fileio.FORMATS:
            back = fileio.read(fileio.write(s, fmt), fmt)
            self.assertEqual(back.formula, "GaN")
            self.assertAlmostEqual(back.volume, s.volume, places=4)
            self.assertAlmostEqual(back.closest_pair()[0], s.closest_pair()[0], places=4)

    def test_density_and_supercell(self):
        au = materials.find(formula="Au")[0].structure()
        self.assertAlmostEqual(au.density, 19.29, places=1)
        self.assertEqual(au.supercell([2, 2, 2]).natoms, 32)


class EnvironmentTests(unittest.TestCase):
    def test_dft_workflow_and_protocol_accuracy(self):
        env = _env(["structure.search", "structure.validate", "dft.prepare", "dft.run", "result.extract"])
        found = env.step({"name": "structure.search", "arguments": {"formula": "SiC"}})
        val = env.step({"name": "structure.validate", "arguments": {"uri": found["candidates"][0]["uri"]}})
        prep = env.step({"name": "dft.prepare", "arguments": {"structure_uri": val["artifact_uri"], "code": "qe-7.2",
                                                              "cutoff_eV": 520, "k_mesh": [6, 6, 6], "relax": True}})
        run = env.step({"name": "dft.run", "arguments": {"input_uri": prep["input_uri"], "max_cpu_hours": 6}})
        ext = env.step({"name": "result.extract", "arguments": {"output_uri": run["output_uri"],
                                                                 "fields": ["energy_per_atom_eV"]}})
        ref = materials.find(formula="SiC")[0].energy_per_atom
        self.assertLess(abs(ext["energy_per_atom_eV"] - ref), 0.002)

    def test_errors_are_typed(self):
        env = _env(["structure.validate", "dft.run"])
        self.assertEqual(env.step({"name": "dft.run", "arguments": {"input_uri": "/etc/passwd", "max_cpu_hours": 1}})["code"],
                         "PATH_OUTSIDE_WORKSPACE")
        self.assertEqual(env.step({"name": "mlp.train", "arguments": {}})["code"], "TOOL_UNAVAILABLE")
        self.assertEqual(env.step({"name": "structure.validate", "arguments": {}})["code"], "INVALID_ARGUMENTS")
        self.assertEqual(env.step({"name": "structure.validate", "arguments": {"uri": "workspace://x.cif"}})["code"],
                         "MISSING_ARTIFACT")

    def test_v2_deprecates_search(self):
        env = _env(["structure.search"], version="v2")
        self.assertEqual(env.step({"name": "structure.search", "arguments": {"formula": "SiC"}})["code"], "DEPRECATED")

    def test_determinism(self):
        a = _env(["structure.search"]).step({"name": "structure.search", "arguments": {"formula": "GaN"}})
        b = _env(["structure.search"]).step({"name": "structure.search", "arguments": {"formula": "GaN"}})
        self.assertEqual(a, b)


class BenchmarkTests(unittest.TestCase):
    def test_split_sizes_and_strata(self):
        test = id_split("test")
        self.assertEqual(len(test), 400)
        counts = {d: sum(t["difficulty"] == d for t in test) for d in ("regular", "missing_data", "tool_failure", "long_horizon")}
        self.assertEqual(counts, {"regular": 200, "missing_data": 80, "tool_failure": 60, "long_horizon": 60})

    def test_no_test_materials_in_training_tasks(self):
        from matact.pipeline.quality import task_materials
        splits, _ = build_benchmark()
        for task in splits["train"]:
            for mid in task_materials(task):
                self.assertEqual(materials.by_id(mid).split, "train", task["task_id"])


if __name__ == "__main__":
    unittest.main()
