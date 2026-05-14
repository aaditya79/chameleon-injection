"""
Global configuration for the Chameleon Injection evaluation framework.

All experiment parameters are set here. Override via environment variables
or by modifying CONFIG directly in experiment scripts.
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class Config:
    # --------------- LLM settings ---------------
    agent_provider: str = "ollama"
    agent_model: str = "llama3.1"
    attacker_provider: str = "ollama"
    attacker_model: str = "llama3.1"
    detector_provider: str = "ollama"
    detector_model: str = "llama3.1"
    judge_provider: str = "ollama"
    judge_model: str = "llama3.1"
    temperature: float = 0.0
    attacker_temperature: float = 0.7

    # --------------- OpenRouter settings ---------------
    openrouter_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", "")
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # --------------- OpenAI / Anthropic ---------------
    openai_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY", "")
    )
    anthropic_api_key: str = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", "")
    )

    # --------------- Experiment settings ---------------
    n_agents: int = 3
    n_rounds: int = 3
    n_camouflage_variants: int = 3
    debate_aggregation: str = "majority_vote"

    # --------------- Scale settings ---------------
    dry_run: bool = True
    max_tasks: Optional[int] = None
    max_trials: Optional[int] = None

    # --------------- Paths ---------------
    results_dir: str = "results"
    data_dir: str = "data"

    # --------------- Cost tracking ---------------
    cost_alert_usd: float = 5.0

    # --------------- Reproducibility ---------------
    seed: int = 42

    # --------------- Second model (comparison / cross-model validation) ---------------
    second_provider: str = "openrouter"
    second_model: str = "mistralai/mistral-7b-instruct:free"
    second_api_key: str = field(
        default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", "")
    )
    run_second_model: bool = False


CONFIG = Config()
