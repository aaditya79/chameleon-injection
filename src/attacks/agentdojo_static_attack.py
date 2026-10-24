"""
AgentDojo adapter for the static-template arm (item 5b). Registers "static": each field a user task exposes is filled
with one fixed overt-injection template from data/static_payloads.json, instantiated with the injection goal. The
template id is ITEM5B_STATIC_TEMPLATE (default static_001, the override-directive baseline). No payload text here.
"""

from __future__ import annotations

import os

from agentdojo.attacks.attack_registry import register_attack
from agentdojo.attacks.base_attacks import BaseAttack
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask


@register_attack
class StaticTemplateAttack(BaseAttack):
    name = "static"

    def attack(self, user_task: BaseUserTask, injection_task: BaseInjectionTask) -> dict[str, str]:
        from src.attacks.static_payloads import instantiate_payloads

        tid = os.environ.get("ITEM5B_STATIC_TEMPLATE", "static_001")
        payloads = {p.payload_id: p for p in instantiate_payloads(injection_task.GOAL)}
        text = payloads[tid].instantiated_text  # fixed template filled with this injection's goal
        return {field: text for field in self.get_injection_candidates(user_task)}
