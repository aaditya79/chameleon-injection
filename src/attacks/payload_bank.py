"""
PayloadBank: manages payload storage and injection delivery.

Handles inserting payloads into clean contexts at three positions:
start, middle (between paragraphs), or end.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.attacks.static_payloads import StaticPayload, instantiate_payloads
from src.attacks.camouflage_generator import CamouflagePayload


# ---------------------------------------------------------------------------
# Injection delivery
# ---------------------------------------------------------------------------

class PayloadBank:
    """
    Central store and delivery mechanism for injection payloads.

    Handles static and camouflage payloads and provides context injection
    at configurable positions.
    """

    def inject(
        self,
        clean_context: str,
        payload_text: str,
        position: str = "end",
    ) -> str:
        """
        Insert payload_text into clean_context at the specified position.

        Args:
            clean_context: The original legitimate document text.
            payload_text: The injection payload to embed.
            position: One of "end", "middle", "start".

        Returns:
            The injected context string.
        """
        if position == "end":
            return clean_context.rstrip() + "\n\n" + payload_text

        if position == "start":
            return payload_text + "\n\n" + clean_context.lstrip()

        if position == "middle":
            return self._inject_middle(clean_context, payload_text)

        raise ValueError(f"Unknown injection position: {position!r}. Use 'end', 'middle', or 'start'.")

    def _inject_middle(self, clean_context: str, payload_text: str) -> str:
        """Split at the midpoint paragraph and insert the payload between halves."""
        paragraphs = [p.strip() for p in clean_context.split("\n\n") if p.strip()]

        if len(paragraphs) <= 1:
            # Fallback: split at sentence midpoint
            sentences = clean_context.split(". ")
            mid = max(1, len(sentences) // 2)
            first = ". ".join(sentences[:mid]) + "."
            second = ". ".join(sentences[mid:])
            return first + "\n\n" + payload_text + "\n\n" + second

        mid = len(paragraphs) // 2
        first_half = "\n\n".join(paragraphs[:mid])
        second_half = "\n\n".join(paragraphs[mid:])
        return first_half + "\n\n" + payload_text + "\n\n" + second_half

    def build_static_payload_text(
        self,
        payload: StaticPayload,
    ) -> str:
        """Return the fully instantiated static payload text."""
        return payload.instantiated_text

    def build_camouflage_payload_text(
        self,
        payload: CamouflagePayload,
    ) -> str:
        """Return the camouflage payload text."""
        return payload.payload_text
