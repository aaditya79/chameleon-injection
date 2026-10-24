"""
AgentDojo adapter for the camouflage attack (item 5b). Registers two attacks that AgentDojo's CLI can run:

    camouflage_full        -- the camouflage text is produced with the surrounding tool output in view
    camouflage_no_context  -- produced without the surrounding tool output (domain/genre/vocab only)

Load into the benchmark with `-ml src.attacks.agentdojo_camouflage_attack`. Each attack only wires AgentDojo's
plumbing (field discovery, goal, surrounding-field text) into ONE text-producing call, `_camouflage_text`, which
is the single function left for the generator author to fill. No injection text is produced in this file; the
author's implementation reads it from the pre-generated data files by id (never inlined here or in chat).
"""

from __future__ import annotations

from agentdojo.attacks.attack_registry import register_attack
from agentdojo.attacks.base_attacks import BaseAttack
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask


class _CamouflageAttack(BaseAttack):
    """Shared plumbing. Subclasses set `use_surrounding_context`; `name` is set on each registered subclass."""

    use_surrounding_context: bool

    def attack(self, user_task: BaseUserTask, injection_task: BaseInjectionTask) -> dict[str, str]:
        # AgentDojo asks, per (user task, injection task), for {injection_field_id: text} covering exactly the
        # fields this user task actually surfaces to the agent. For each such field we hand the generator the
        # injection goal and (for the full variant) that field's default benign content as the register to match.
        defaults = self.task_suite.get_injection_vector_defaults()
        out: dict[str, str] = {}
        for field_id in self.get_injection_candidates(user_task):
            out[field_id] = self._camouflage_text(
                goal=injection_task.GOAL,
                surrounding=defaults[field_id] if self.use_surrounding_context else None,
                field_id=field_id,
                user_task_id=user_task.ID,
                injection_task_id=injection_task.ID,
            )
        return out

    domain: str = "financial"  # banking suite -> financial domain for the generator / domains.json

    def _camouflage_text(
        self,
        *,
        goal: str,              # injection_task.GOAL: the malicious goal to steer toward, as AgentDojo states it
        surrounding: str | None,  # the field's default benign tool-output text to match in register; None for no_context
        field_id: str,          # which injection vector in this suite (e.g. a bill / notice field)
        user_task_id: str,      # unused at matrix time (payloads are keyed by arm/injection_task/field)
        injection_task_id: str,
    ) -> str:                   # returns the pre-generated, screened camouflage text for this slot
        """Return the pre-generated screened camouflage text for this slot, or raise NoScreenedPayload.

        Delegates to camouflage_screened.screened_payload with generate=False: at matrix time it only reads the
        payload the pre-step wrote to data/item5b_camouflage_payloads.json (keyed by arm/injection_task/field), so
        episodes are deterministic and generate nothing. The pre-step (experiments/item5b_gen_payloads.py) is what
        runs the CamouflageGenerator and the refusal screen and writes those payloads by id.
        """
        from src.attacks.camouflage_screened import screened_payload

        return screened_payload(self.name, goal, surrounding, field_id, injection_task_id,
                                domain=self.domain, generate=False)


class NoScreenedPayload(Exception):
    """Raise from `_camouflage_text` when no screened payload exists for a slot; the runner excludes the pair."""


@register_attack
class CamouflageFullAttack(_CamouflageAttack):
    name = "camouflage_full"
    use_surrounding_context = True


@register_attack
class CamouflageNoContextAttack(_CamouflageAttack):
    name = "camouflage_no_context"
    use_surrounding_context = False
