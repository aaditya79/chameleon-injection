"""
Static payload bank: loads and instantiates static injection payloads
from data/static_payloads.json, filling in the malicious_goal for each task.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class StaticPayload:
    """One instantiated static payload with the malicious goal filled in."""

    payload_id: str
    category: str
    template: str
    instantiated_text: str
    malicious_goal: str
    obviousness_score: int


def load_static_payload_templates(data_dir: str = "data") -> List[dict]:
    """Load raw payload templates from disk."""
    path = Path(data_dir) / "static_payloads.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def instantiate_payloads(
    malicious_goal: str,
    data_dir: str = "data",
    categories: Optional[List[str]] = None,
) -> List[StaticPayload]:
    """
    Instantiate all (or filtered) static payload templates with a specific malicious goal.

    Args:
        malicious_goal: The attacker's desired outcome, inserted into each template.
        data_dir: Directory containing static_payloads.json.
        categories: If provided, only return payloads in these categories.

    Returns:
        List of instantiated StaticPayload objects.
    """
    templates = load_static_payload_templates(data_dir)
    payloads = []
    for tmpl in templates:
        if categories and tmpl["category"] not in categories:
            continue
        instantiated = tmpl["template"].replace("{malicious_goal}", malicious_goal)
        payloads.append(
            StaticPayload(
                payload_id=tmpl["payload_id"],
                category=tmpl["category"],
                template=tmpl["template"],
                instantiated_text=instantiated,
                malicious_goal=malicious_goal,
                obviousness_score=tmpl["obviousness_score"],
            )
        )
    return payloads
