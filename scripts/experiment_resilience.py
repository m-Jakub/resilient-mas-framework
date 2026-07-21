"""
Utilities for resilient experiment execution (JSONL progress + resume support).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass
class ExperimentProgressLogger:
    progress_path: Path

    def get_or_create_run_id(self, run_prefix: str) -> str:
        records = _read_jsonl(self.progress_path)
        last_run_id: Optional[str] = None
        completed_runs: Set[str] = set()

        for rec in records:
            if rec.get("type") == "run_start":
                last_run_id = rec.get("run_id")
            elif rec.get("type") == "run_complete":
                completed_runs.add(rec.get("run_id"))

        if last_run_id and last_run_id not in completed_runs:
            return last_run_id

        return f"{run_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def ensure_run_started(self, run_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        records = _read_jsonl(self.progress_path)
        if any(rec.get("type") == "run_start" and rec.get("run_id") == run_id for rec in records):
            return

        _append_jsonl(
            self.progress_path,
            {
                "type": "run_start",
                "run_id": run_id,
                "timestamp": _utcnow_iso(),
                "metadata": metadata or {},
            },
        )

    def load_completed_experiment_ids(self, run_id: str) -> Set[str]:
        completed: Set[str] = set()
        for rec in _read_jsonl(self.progress_path):
            if (
                rec.get("type") == "experiment_result"
                and rec.get("run_id") == run_id
                and rec.get("status") == "success"
            ):
                exp_id = rec.get("experiment_id")
                if exp_id:
                    completed.add(exp_id)
        return completed

    def log_experiment_result(self, run_id: str, result: Dict[str, Any]) -> None:
        record = {
            "type": "experiment_result",
            "run_id": run_id,
            "timestamp": _utcnow_iso(),
            **result,
        }
        _append_jsonl(self.progress_path, record)

    def log_run_complete(self, run_id: str, summary: Optional[Dict[str, Any]] = None) -> None:
        _append_jsonl(
            self.progress_path,
            {
                "type": "run_complete",
                "run_id": run_id,
                "timestamp": _utcnow_iso(),
                "summary": summary or {},
            },
        )


def initialize_experiment_run(
    progress_path: Path,
    run_prefix: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> tuple[ExperimentProgressLogger, str, Set[str]]:
    """
    Template helper for new runners:
    - ensures run_start
    - returns completed experiment IDs for resume

    Usage:
        logger, run_id, completed_ids = initialize_experiment_run(
            Path("logs/experiments/progress/my_experiment.jsonl"),
            "my_experiment",
            metadata={"note": "batch-1"},
        )
    """
    logger = ExperimentProgressLogger(progress_path)
    run_id = logger.get_or_create_run_id(run_prefix)
    logger.ensure_run_started(run_id, metadata=metadata)
    completed_ids = logger.load_completed_experiment_ids(run_id)
    return logger, run_id, completed_ids
