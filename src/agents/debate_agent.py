"""
DebateAgent: a single participant in a multi-agent debate.

Each agent sees the context, generates a position in round 1, then
reviews peer positions and updates its stance in subsequent rounds.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from src.utils.llm_client import LLMClient, CompletionResult
from src.agents.single_agent import AgentResponse


# ---------------------------------------------------------------------------
# Debate round dataclass
# ---------------------------------------------------------------------------

@dataclass
class DebateRound:
    """All agent responses for one round of debate."""

    round_num: int
    responses: List[AgentResponse]

    def format_for_history(self) -> str:
        """Format this round as a readable history block."""
        lines = [f"=== Round {self.round_num} ==="]
        for r in self.responses:
            lines.append(f"{r.agent_id}: {r.response_text[:600]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_ROUND_1 = """\
You are {role}, a professional analyst participating in a multi-analyst review. \
Analyze the provided document carefully and state your position clearly. \
Be specific and cite evidence from the document. Do not simply agree with others \
— form an independent judgment based on the document.\
"""

_SYSTEM_PROMPT_LATER_ROUNDS = """\
You are {role}. Review the positions stated by your colleagues in the previous \
round below. Consider their arguments critically. You may update your position \
if persuaded by strong evidence, but do not simply defer to others — challenge \
weak reasoning and defend your position if you believe it is correct.

PREVIOUS ROUND POSITIONS:
{debate_history}

Now state your updated position for round {round_num}. Be concise and direct.\
"""

_USER_TEMPLATE = """\
TASK: {instruction}

DOCUMENT:
{context}\
"""


class DebateAgent:
    """
    A single agent in a multi-agent debate.

    Args:
        agent_id: Unique identifier for this agent (e.g., "Analyst_A").
        role: Role label used in system prompts.
        llm_client: LLMClient instance.
    """

    def __init__(
        self,
        agent_id: str,
        role: str,
        llm_client: LLMClient,
    ) -> None:
        self.agent_id = agent_id
        self.role = role
        self.client = llm_client

    def generate_position(
        self,
        instruction: str,
        context: str,
        debate_history: List[DebateRound],
        round_num: int,
    ) -> AgentResponse:
        """
        Generate this agent's position for the current debate round.

        Args:
            instruction: The task instruction.
            context: The document context (may be injected).
            debate_history: All completed rounds so far.
            round_num: Current round number (1-indexed).

        Returns:
            AgentResponse with this agent's position text.
        """
        if round_num == 1:
            system = _SYSTEM_PROMPT_ROUND_1.format(role=self.role)
        else:
            history_text = "\n\n".join(
                r.format_for_history() for r in debate_history
            )
            system = _SYSTEM_PROMPT_LATER_ROUNDS.format(
                role=self.role,
                debate_history=history_text,
                round_num=round_num,
            )

        user = _USER_TEMPLATE.format(instruction=instruction, context=context)

        completion = self.client.complete(
            system=system,
            user=user,
            temperature=0.0,
            max_tokens=500,
        )
        return AgentResponse(
            agent_id=self.agent_id,
            response_text=completion.content,
            completion=completion,
        )
