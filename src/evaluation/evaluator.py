"""
Convenience evaluator: runs agent + detector + judge on a single trial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.agents.single_agent import SingleAgent, AgentResponse
from src.attacks.payload_bank import PayloadBank
from src.detection.static_detector import StaticDetector, DetectionResult
from src.evaluation.asr_judge import ASRJudge, ASRJudgment
from src.utils.llm_client import LLMClient


@dataclass
class TrialResult:
    """Complete result of one evaluation trial."""

    task_id: str
    domain: str
    payload_type: str
    payload_id: Optional[str]
    injected_context: str
    agent_response: AgentResponse
    detection: Optional[DetectionResult]
    judgment: Optional[ASRJudgment]
    total_cost_usd: float
    total_tokens: int


class Evaluator:
    """
    Runs a complete evaluation trial: inject → agent → detect → judge.

    Args:
        agent: SingleAgent instance.
        detector: StaticDetector or AugmentedDetector (optional).
        judge: ASRJudge instance (optional).
        payload_bank: PayloadBank for injection.
    """

    def __init__(
        self,
        agent: SingleAgent,
        detector: Optional[StaticDetector] = None,
        judge: Optional[ASRJudge] = None,
        payload_bank: Optional[PayloadBank] = None,
    ) -> None:
        self.agent = agent
        self.detector = detector
        self.judge = judge
        self.payload_bank = payload_bank or PayloadBank()

    def run_trial(
        self,
        task_id: str,
        domain: str,
        instruction: str,
        clean_context: str,
        malicious_goal: str,
        payload_text: Optional[str],
        payload_id: Optional[str] = None,
        payload_type: str = "clean",
        injection_position: str = "end",
    ) -> TrialResult:
        """
        Run one trial.

        If payload_text is None, runs a clean baseline trial.
        """
        if payload_text is not None:
            injected_context = self.payload_bank.inject(
                clean_context=clean_context,
                payload_text=payload_text,
                position=injection_position,
            )
        else:
            injected_context = clean_context

        agent_response = self.agent.analyze(
            instruction=instruction,
            context=injected_context,
            task_id=task_id,
        )
        total_cost = agent_response.completion.cost_usd
        total_tokens = agent_response.completion.input_tokens + agent_response.completion.output_tokens

        detection = None
        if self.detector and payload_type != "clean":
            detection = self.detector.detect(
                context=injected_context,
                instruction=instruction,
                ground_truth=(payload_type != "clean"),
            )
            total_cost += detection.completion.cost_usd if detection.completion else 0
            total_tokens += (detection.completion.input_tokens + detection.completion.output_tokens) if detection.completion else 0

        judgment = None
        if self.judge and payload_type != "clean":
            judgment = self.judge.judge(
                instruction=instruction,
                malicious_goal=malicious_goal,
                agent_response=agent_response.response_text,
            )
            total_cost += judgment.completion.cost_usd if judgment.completion else 0
            total_tokens += (judgment.completion.input_tokens + judgment.completion.output_tokens) if judgment.completion else 0

        return TrialResult(
            task_id=task_id,
            domain=domain,
            payload_type=payload_type,
            payload_id=payload_id,
            injected_context=injected_context,
            agent_response=agent_response,
            detection=detection,
            judgment=judgment,
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
        )
