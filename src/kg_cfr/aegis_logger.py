"""
AEGIS logger for Tripartite Standoff sessions.

Records module turns, shocks, and summary statistics without team framing.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import csv
import json


@dataclass
class ModuleTurnLog:
    turn_number: int
    timestamp: str
    phase: str
    module_id: str
    module_label: str
    message_content: str
    message_length: int
    concepts_mentioned: List[str] = field(default_factory=list)
    response_time_ms: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShockLog:
    timestamp: str
    shock_type: str
    shock_text: str
    turn_number: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class AegisLogger:
    def __init__(self, session_id: str, crisis_context: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.session_id = session_id
        self.crisis_context = crisis_context
        self.session_start = datetime.now().isoformat()
        self.session_metadata = metadata or {}
        self.turn_logs: List[ModuleTurnLog] = []
        self.shock_logs: List[ShockLog] = []
        self.stats: Dict[str, Any] = {}

    def log_turn(
        self,
        turn_number: int,
        phase: str,
        module_id: str,
        module_label: str,
        message_content: str,
        concepts_mentioned: Optional[List[str]] = None,
        response_time_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        concepts = concepts_mentioned or []
        log = ModuleTurnLog(
            turn_number=turn_number,
            timestamp=datetime.now().isoformat(),
            phase=phase,
            module_id=module_id,
            module_label=module_label,
            message_content=message_content,
            message_length=len(message_content or ""),
            concepts_mentioned=concepts,
            response_time_ms=response_time_ms,
            metadata=metadata or {},
        )
        self.turn_logs.append(log)

    def log_shock(self, shock_type: str, shock_text: str, turn_number: int, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.shock_logs.append(
            ShockLog(
                timestamp=datetime.now().isoformat(),
                shock_type=shock_type,
                shock_text=shock_text,
                turn_number=turn_number,
                metadata=metadata or {},
            )
        )

    def finalize(self) -> None:
        self.stats = {
            "total_turns": len(self.turn_logs),
            "total_shocks": len(self.shock_logs),
        }

        phase4_turns = [t for t in self.turn_logs if t.phase == "phase4_synthesis"]
        if not phase4_turns:
            return

        pf_scores: List[float] = []
        token_overhead_total = 0
        self_correction_counts = 0
        self_correction_rates: List[float] = []
        iterations: List[int] = []
        latency_ms_values: List[int] = []

        for turn in phase4_turns:
            meta = turn.metadata or {}
            pf = meta.get("avg_provenance_fidelity")
            if pf is None:
                pf = meta.get("provenance_fidelity")
            if isinstance(pf, (int, float)):
                pf_scores.append(float(pf))

            token_overhead_total += int(meta.get("total_token_overhead", 0) or 0)
            self_correction_counts += int(meta.get("self_correction_count", 0) or 0)

            scr = meta.get("self_correction_rate")
            if isinstance(scr, (int, float)):
                self_correction_rates.append(float(scr))

            iteration_count = meta.get("synthesis_iterations")
            if isinstance(iteration_count, int):
                iterations.append(iteration_count)

            latency_ms = meta.get("phase4_execution_latency_ms")
            if isinstance(latency_ms, (int, float)):
                latency_ms_values.append(int(latency_ms))

        avg_pf = sum(pf_scores) / len(pf_scores) if pf_scores else 0.0
        avg_scr = sum(self_correction_rates) / len(self_correction_rates) if self_correction_rates else 0.0
        avg_iterations = sum(iterations) / len(iterations) if iterations else 0.0
        avg_latency_ms = sum(latency_ms_values) / len(latency_ms_values) if latency_ms_values else 0.0
        total_latency_ms = sum(latency_ms_values) if latency_ms_values else 0

        self.stats.update({
            "phase4_runs": len(phase4_turns),
            "phase4_avg_provenance_fidelity": avg_pf,
            "phase4_total_token_overhead": token_overhead_total,
            "phase4_self_correction_count": self_correction_counts,
            "phase4_avg_self_correction_rate": avg_scr,
            "phase4_avg_iterations": avg_iterations,
            "phase4_avg_execution_latency_ms": avg_latency_ms,
            "phase4_total_execution_latency_ms": total_latency_ms,
        })

    def save_json(self, output_path: Path) -> None:
        payload = {
            "session_info": {
                "session_id": self.session_id,
                "crisis_context": self.crisis_context,
                "session_start": self.session_start,
                "metadata": self.session_metadata,
                "statistics": self.stats,
            },
            "turn_logs": [asdict(t) for t in self.turn_logs],
            "shock_logs": [asdict(s) for s in self.shock_logs],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def save_csv(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.turn_logs:
            turns_path = output_dir / f"{self.session_id}_turns.csv"
            with turns_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(asdict(self.turn_logs[0]).keys()))
                writer.writeheader()
                for row in self.turn_logs:
                    writer.writerow(asdict(row))

        if self.shock_logs:
            shocks_path = output_dir / f"{self.session_id}_shocks.csv"
            with shocks_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(asdict(self.shock_logs[0]).keys()))
                writer.writeheader()
                for row in self.shock_logs:
                    writer.writerow(asdict(row))
