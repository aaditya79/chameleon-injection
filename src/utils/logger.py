"""
Trial logger: writes every experiment trial to results/trials.jsonl
and maintains aggregate metrics snapshots.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class TrialLogger:
    """
    Append-only JSONL logger for experiment trials.

    Each call to log_trial() appends one JSON line to trials.jsonl.
    Metrics snapshots are saved separately to metrics_summary.json.

    Args:
        results_dir: Directory where output files are written.
    """

    def __init__(self, results_dir: str = "results") -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.trials_path = self.results_dir / "trials.jsonl"
        self.metrics_path = self.results_dir / "metrics_summary.json"
        self._trial_count = 0

    def log_trial(
        self,
        *,
        experiment: str,
        task_id: str,
        domain: str,
        architecture: str,
        payload_type: str,
        payload_id: Optional[str] = None,
        payload_category: Optional[str] = None,
        injection_position: str = "end",
        injected_agent_idx: Optional[int] = None,
        agent_responses: list[str],
        debate_history: Optional[list[dict]] = None,
        final_answer: str,
        asr_judgment: Optional[dict] = None,
        detection_result: Optional[dict] = None,
        semantic_similarity: Optional[float] = None,
        total_tokens: int = 0,
        total_cost_usd: float = 0.0,
        model: str = "",
        provider: str = "",
        extra: Optional[dict] = None,
    ) -> str:
        """
        Write one trial record to trials.jsonl.

        Returns:
            The trial_id assigned to this record.
        """
        trial_id = str(uuid.uuid4())
        record: dict[str, Any] = {
            "trial_id": trial_id,
            "experiment": experiment,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": task_id,
            "domain": domain,
            "architecture": architecture,
            "payload_type": payload_type,
            "payload_id": payload_id,
            "payload_category": payload_category,
            "injection_position": injection_position,
            "injected_agent_idx": injected_agent_idx,
            "agent_responses": agent_responses,
            "debate_history": debate_history,
            "final_answer": final_answer,
            "asr_judgment": asr_judgment,
            "detection_result": detection_result,
            "semantic_similarity": semantic_similarity,
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost_usd,
            "model": model,
            "provider": provider,
        }
        if extra:
            record.update(extra)

        with open(self.trials_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        self._trial_count += 1
        return trial_id

    def save_metrics_snapshot(self, metrics: dict, experiment: str) -> None:
        """
        Merge a metrics dict into metrics_summary.json under the given experiment key.
        """
        existing: dict = {}
        if self.metrics_path.exists():
            with open(self.metrics_path, encoding="utf-8") as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    existing = {}

        existing[experiment] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
        }
        with open(self.metrics_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)

    def load_trials(self, experiment: Optional[str] = None) -> list[dict]:
        """
        Load all trials from trials.jsonl, optionally filtered by experiment.
        """
        if not self.trials_path.exists():
            return []
        records = []
        with open(self.trials_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if experiment is None or rec.get("experiment") == experiment:
                        records.append(rec)
                except json.JSONDecodeError:
                    continue
        return records

    def load_completed_signatures(self) -> set[tuple]:
        """
        Read trials.jsonl and return the set of completed trial signatures.

        Each signature is a 6-tuple:
            (task_id, payload_type, payload_id, architecture, inject_mode, model)

        payload_id defaults to "" for null values.
        inject_mode defaults to "none" for records written before resume logic was added.

        Use this for resume logic: before running a trial, check whether its
        signature already appears in this set, and skip if so.
        """
        sigs: set[tuple] = set()
        for rec in self.load_trials():
            sig = (
                rec.get("task_id", ""),
                rec.get("payload_type", ""),
                rec.get("payload_id") or "",
                rec.get("architecture", ""),
                rec.get("inject_mode", "none"),
                rec.get("model", ""),
            )
            sigs.add(sig)
        return sigs

    @property
    def trial_count(self) -> int:
        """Number of trials logged in this session."""
        return self._trial_count
