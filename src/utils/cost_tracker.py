"""
Global cost tracker: aggregates cost across multiple LLMClient instances.
"""

from __future__ import annotations

import threading
from typing import Optional


class CostTracker:
    """
    Thread-safe global cost accumulator.

    Use as a singleton (COST_TRACKER) to track spend across all LLMClient
    instances in an experiment run.

    Args:
        alert_threshold_usd: Print a warning each time this threshold is crossed.
    """

    def __init__(self, alert_threshold_usd: float = 5.0) -> None:
        self._lock = threading.Lock()
        self._total_cost: float = 0.0
        self._total_tokens: int = 0
        self._call_count: int = 0
        self.alert_threshold_usd = alert_threshold_usd
        self._last_alert_at: float = 0.0

    def record(self, cost_usd: float, tokens: int = 0) -> None:
        """Add cost and token counts from one completion."""
        with self._lock:
            self._total_cost += cost_usd
            self._total_tokens += tokens
            self._call_count += 1
            if (
                self._total_cost >= self.alert_threshold_usd
                and self._total_cost - cost_usd < self.alert_threshold_usd
            ):
                print(
                    f"\n[COST ALERT] Total spend crossed ${self.alert_threshold_usd:.2f}. "
                    f"Current total: ${self._total_cost:.4f}\n"
                )

    def checkpoint(self, label: str = "") -> None:
        """Print current totals. Call every N trials."""
        with self._lock:
            tag = f" ({label})" if label else ""
            print(
                f"[Cost checkpoint{tag}] "
                f"${self._total_cost:.4f} | "
                f"{self._total_tokens:,} tokens | "
                f"{self._call_count} calls"
            )

    @property
    def total_cost(self) -> float:
        """Total cost in USD."""
        with self._lock:
            return self._total_cost

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed."""
        with self._lock:
            return self._total_tokens

    @property
    def call_count(self) -> int:
        """Total number of LLM calls tracked."""
        with self._lock:
            return self._call_count

    def reset(self) -> None:
        """Reset all counters."""
        with self._lock:
            self._total_cost = 0.0
            self._total_tokens = 0
            self._call_count = 0
            self._last_alert_at = 0.0


# Module-level singleton
COST_TRACKER = CostTracker()
