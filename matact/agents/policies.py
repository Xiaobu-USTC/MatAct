"""Policies: scripted expert workflows and the model-backed policy interface."""
from __future__ import annotations

from ..protocol import format_answer, format_call
from . import workflows_t12 as w12
from . import workflows_t34 as w34
from .base import Call, Context, Style

WORKFLOWS = {
    "retrieve_convert": w12.t1_retrieve, "retrieve_missing": w12.t1_retrieve, "file_convert": w12.t1_file,
    "repair_convert": w12.t1_file, "wrong_path": w12.t1_file, "density_compare": w12.t1_density,
    "dft_property": w12.t2_property, "dft_missing": w12.t2_property, "dft_scf_repair": w12.t2_property,
    "dft_timeout": w12.t2_property, "dft_singlepoint": w12.t2_singlepoint, "dft_mismatch": w12.t2_mismatch,
    "mlp_finetune": w34.t3_finetune, "mlp_missing": w34.t3_finetune, "mlp_unstable_repair": w34.t3_finetune,
    "mlp_compare": w34.t3_finetune, "mlp_pretrained": w34.t3_pretrained,
    "db_lookup": w34.t4_lookup, "db_missing": w34.t4_lookup, "db_filter": w34.t4_filter,
    "db_insert": w34.t4_insert, "db_insert_repair": w34.t4_insert, "db_insert_two": w34.t4_insert,
    "compute_register": w34.c_compute_register, "db_fallback_dft": w34.c_db_fallback,
}


class Policy:
    """Interface used by the rollout runner. ``act`` returns the assistant text for one turn."""
    name = "policy"

    def start(self, task: dict, state: dict):
        pass

    def act(self, messages: list, observation) -> str:
        raise NotImplementedError


class ScriptedPolicy(Policy):
    """Expert workflow policy. It reads the public task parameters and observations only."""
    name = "expert"

    def __init__(self, style: Style = None):
        self.style = style or Style()

    def start(self, task, state):
        self.ctx = Context(task, state, self.style)
        self.gen = WORKFLOWS[task["subtype"]](self.ctx, task["params"])
        self.started = False

    def next_item(self, observation):
        """Advance the workflow; returns a Call or an Answer."""
        try:
            item = self.gen.send(observation) if self.started else next(self.gen)
        except StopIteration as stop:
            return stop.value
        self.started = True
        return item

    def act(self, messages, observation):
        item = self.next_item(observation)
        if isinstance(item, Call):
            return format_call(item.name, item.arguments, item.rationale)
        return format_answer(item.text, item.rationale)
