import copy
import json
import random
import unittest

from matact.agents.base import Style
from matact.agents.policies import ScriptedPolicy
from matact.integrations.verl_reward import compute_score, compute_score_generic
from matact.pipeline.export import to_rl_row
from matact.pipeline.normalize import Rejected, normalize
from matact.pipeline.quality import assess, dedup_exact
from matact.pipeline.sources import inject_defect, to_raw
from matact.rollout import run_episode
from matact.tasks.generator import id_split
from matact.train.weighted_sft import WEIGHTS, char_weights, token_weights
from matact.validator import RuleValidator
from matact.verify.scoring import score

TASKS = {t["subtype"]: t for t in id_split("valid")}


class PipelineTests(unittest.TestCase):
    def test_legacy_names_and_multi_action_messages_are_normalized(self):
        task = TASKS["mlp_finetune"]
        record, _ = run_episode(task, ScriptedPolicy())
        raw = to_raw(record, "t-1", "agent", task["task_id"], random.Random(0), legacy=True, p_merge=1.0)
        self.assertTrue(any(m["content"].count("<tool_call>") > 1 for m in raw["messages"]))
        traj = normalize(raw)
        self.assertEqual([s["action"] for s in traj["steps"]], [s["action"] for s in record["steps"]])
        sc, reasons = assess(task, dict(traj, scores={}))
        self.assertEqual(reasons, [])
        self.assertGreaterEqual(sc["utility"], 0.72)

    def test_defects_are_rejected(self):
        task = TASKS["dft_property"]
        record, _ = run_episode(task, ScriptedPolicy())
        for kind, expected in (("missing_observation", "missing observation"), ("interrupted", "missing")):
            raw = to_raw(record, "t-2", "expert", task["task_id"], random.Random(1))
            inject_defect(raw, kind, random.Random(1))
            with self.assertRaises(Rejected) as ctx:
                normalize(raw)
            self.assertIn(expected, str(ctx.exception))
        raw = to_raw(record, "t-3", "expert", task["task_id"], random.Random(1))
        inject_defect(raw, "unsupported_claim", random.Random(1))
        _, reasons = assess(task, normalize(raw))
        self.assertIn("unsupported conclusion", reasons)

    def test_dedup_keeps_best(self):
        a = {"trajectory_id": "a", "task_id": "x", "steps": [], "answer": "y", "scores": {"utility": 0.8, "cost": 5}}
        b = dict(copy.deepcopy(a), trajectory_id="b", scores={"utility": 0.9, "cost": 5})
        kept, dropped = dedup_exact([a, b])
        self.assertEqual([t["trajectory_id"] for t in kept], ["b"])
        self.assertEqual(dropped, ["a"])


class InventoryTests(unittest.TestCase):
    def test_released_inventory_matches_the_paper(self):
        from matact.env.handlers_compute import K_DENSITY
        from matact.pipeline.sources import SOURCE_RETAINED, SOURCE_SIZES
        self.assertEqual(sum(SOURCE_SIZES.values()), 7100)
        self.assertEqual(SOURCE_RETAINED, {"expert": 1260, "agent": 1180, "repair": 840, "synthetic": 1620})
        total = sum(SOURCE_RETAINED.values())
        self.assertEqual(total, 4900)
        shares = [round(100 * SOURCE_RETAINED[s] / total, 1) for s in ("expert", "agent", "repair", "synthetic")]
        self.assertEqual(shares, [25.7, 24.1, 17.1, 33.1])
        self.assertAlmostEqual(K_DENSITY, 2 * 3.141592653589793 / 0.15, places=9)

    def test_inventory_selection(self):
        from matact.pipeline.seed_pool import select_inventory
        candidates = [{"trajectory_id": f"{s}-{i:03d}", "source": s}
                      for s in ("expert", "agent", "repair", "synthetic") for i in range(5)]
        report = [{"trace_id": c["trajectory_id"], "retained": True, "reasons": []} for c in candidates]
        chosen = select_inventory(candidates, report, seed=0,
                                  targets={"expert": 2, "agent": 2, "repair": 2, "synthetic": 2})
        self.assertEqual(len(chosen), 8)
        self.assertEqual(sorted(t["source"] for t in chosen),
                         ["agent", "agent", "expert", "expert", "repair", "repair", "synthetic", "synthetic"])
        self.assertEqual(sum(e["retained"] for e in report), 8)
        self.assertTrue(all("duplicate" in e["reasons"][-1] for e in report if not e["retained"]))


class VerifierTests(unittest.TestCase):
    def test_expert_solves_every_subtype(self):
        for subtype, task in TASKS.items():
            record, env = run_episode(task, ScriptedPolicy())
            sc = score(task, record, env)
            self.assertTrue(sc["success"] and sc["factual"], subtype)

    def test_protocol_violation_fails_tier_b(self):
        task = next(t for t in id_split("valid") if t["subtype"] == "dft_property"
                    and t["params"]["property"] != "band_gap_eV")
        record, env = run_episode(task, ScriptedPolicy(Style(cutoff=300.0, k_scale=0.4)))
        sc = score(task, record, env)
        self.assertFalse(sc["checks"]["protocol"])
        self.assertFalse(sc["success"])

    def test_verl_reward_replays_actions(self):
        task = next(t for t in id_split("valid") if t["subtype"] == "db_insert")
        record, _ = run_episode(task, ScriptedPolicy())
        text = "\n".join(m["content"] for m in record["messages"][2:] if m["role"] == "assistant")
        good = compute_score("matact", text, json.dumps(task))
        bad = compute_score("matact", "<answer>The value is 1.234 eV.</answer>", json.dumps(task))
        self.assertGreater(good["score"], 0.8)
        self.assertEqual(good["success"], 1.0)
        self.assertLess(bad["score"], 0.3)
        generic = compute_score_generic("matact", text, json.dumps(task))
        self.assertAlmostEqual(generic["score"], 1.0, places=6)  # every call of the expert succeeds

    def test_rl_rows_share_one_schema(self):
        rows = [to_rl_row(t, i) for i, t in enumerate(TASKS.values())]
        keys = {json.dumps({k: sorted(v["create_kwargs"]) for k, v in r["extra_info"]["tools_kwargs"].items()})
                for r in rows}
        self.assertEqual(len(keys), 1)
        self.assertTrue(all(sum(bool(v["create_kwargs"]["env_spec"]) for v in r["extra_info"]["tools_kwargs"].values()) == 1
                            for r in rows))

    def test_format_compliance_requires_schema_valid_calls(self):
        task = next(t for t in id_split("valid") if t["subtype"] == "db_insert")
        record, env = run_episode(task, ScriptedPolicy())
        sc = score(task, record, env)
        self.assertEqual(sc["n_format_ok"], sc["n_turns"])
        record_bad = {"task_id": task["task_id"], "status": "answer", "steps": record["steps"] +
                      [{"action": {"name": "database.update", "arguments": {"database": "matact_userdb",
                                                                            "mode": "insert", "record": []}},
                        "observation": {"observation_id": "obs-x", "status": "error", "code": "INVALID_ARGUMENTS"}}],
                      "answer": record["answer"]}
        bad = score(task, record_bad, env)
        self.assertEqual(bad["n_format_ok"], sc["n_turns"])
        self.assertLess(bad["n_format_ok"], bad["n_turns"])


class WeightingTests(unittest.TestCase):
    def test_action_weights(self):
        messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "do it"},
                    {"role": "assistant", "content": 'Plan.\n<tool_call>\n{"name": "dft.run", "arguments": {"x": 1}}\n</tool_call>'},
                    {"role": "tool", "content": '{"status": "success"}'},
                    {"role": "assistant", "content": "<answer>\nE = 1 eV\n</answer>"}]
        rendered = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        w = char_weights(rendered, messages)
        at = lambda s: w[rendered.index(s)]  # noqa: E731
        self.assertEqual(at("SYS"), 0.0)
        self.assertEqual(at("do it"), WEIGHTS["user"])
        self.assertEqual(at("Plan."), WEIGHTS["rationale"])
        self.assertEqual(at("dft.run"), WEIGHTS["tool_name"])
        self.assertEqual(at('"x"'), WEIGHTS["argument"])
        self.assertEqual(at("<tool_call>"), WEIGHTS["boundary"])
        self.assertEqual(at('{"status"'), 0.0)
        self.assertEqual(at("E = 1"), WEIGHTS["answer"])
        offsets = [(0, 3), (rendered.index("dft.run"), rendered.index("dft.run") + 3), (5, 5)]
        self.assertEqual(token_weights(w, offsets), [0.0, WEIGHTS["tool_name"], 0.0])


if __name__ == "__main__":
    unittest.main()
