"""
Task generator: uses an LLM to generate additional task contexts if needed.
The 45 tasks in data/tasks.json are hand-crafted and do not require this module
for the main experiments.
"""

from __future__ import annotations

from src.utils.llm_client import LLMClient


class TaskGenerator:
    """
    Generates synthetic task contexts for data augmentation.

    Not used in the main experimental pipeline — tasks.json is the canonical source.

    Args:
        llm_client: LLMClient configured for the task generation model.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self.client = llm_client

    def generate_context(self, domain: str, subcategory: str, task_description: str) -> str:
        """Generate a synthetic clean context for a given task description."""
        system = (
            f"You are generating synthetic {domain} documents for research. "
            "Write a realistic, professional document excerpt in the appropriate register."
        )
        user = (
            f"Write a 200-350 word {domain} document excerpt for a task in the "
            f"'{subcategory}' subcategory. Task: {task_description}. "
            "Use domain-appropriate vocabulary and structure. "
            "Do NOT include any injection or manipulation text."
        )
        result = self.client.complete(system=system, user=user, max_tokens=500)
        return result.content.strip()
