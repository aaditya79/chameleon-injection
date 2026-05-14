"""
Task bank utilities: load and filter tasks from data/tasks.json.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional


def load_tasks(
    data_dir: str = "data",
    domains: Optional[List[str]] = None,
    max_tasks: Optional[int] = None,
    seed: int = 42,
) -> List[dict]:
    """
    Load tasks from disk with optional domain filtering and count limit.

    Args:
        data_dir: Directory containing tasks.json.
        domains: If provided, only return tasks in these domains.
        max_tasks: Maximum number of tasks to return.
        seed: Random seed for sampling when max_tasks < total.

    Returns:
        List of task dicts.
    """
    path = Path(data_dir) / "tasks.json"
    with open(path, encoding="utf-8") as f:
        tasks = json.load(f)

    if domains:
        tasks = [t for t in tasks if t.get("domain") in domains]

    if max_tasks is not None and max_tasks < len(tasks):
        rng = random.Random(seed)
        tasks = rng.sample(tasks, max_tasks)

    return tasks


def get_task_by_id(task_id: str, data_dir: str = "data") -> Optional[dict]:
    """Return a single task by its task_id."""
    tasks = load_tasks(data_dir)
    for t in tasks:
        if t["task_id"] == task_id:
            return t
    return None
