"""
Export a readable debate transcript from JSON logs.

Usage:
  python scripts/analysis/export_transcript.py --input logs/production/debate_20260203_134032.json --output logs/production/transcripts/debate_20260203_134032.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_header(f, session_info: Dict[str, Any]) -> None:
    stats = session_info.get("statistics", {})
    meta = session_info.get("metadata", {})
    f.write("=" * 80 + "\n")
    f.write("PHILOSOPHICAL DEBATE TRANSCRIPT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Session ID: {_safe_text(session_info.get('session_id', 'unknown'))}\n")
    f.write(f"Topic: {_safe_text(session_info.get('topic', ''))}\n")
    f.write(f"Start: {_safe_text(session_info.get('session_start', ''))}\n")
    f.write(f"Duration (s): {_safe_text(stats.get('session_duration_seconds', ''))}\n")
    f.write(f"Total turns: {_safe_text(stats.get('total_turns', ''))}\n")
    f.write(f"Moderation enabled: {_safe_text(meta.get('moderation_enabled', ''))}\n")
    f.write(f"ToM enabled: {_safe_text(meta.get('tom_enabled', ''))}\n")
    f.write("\n")


def _write_moderation(f, moderation_logs: List[Dict[str, Any]]) -> None:
    f.write("=" * 80 + "\n")
    f.write("PHASE: SOCRATIC MODERATION\n")
    f.write("=" * 80 + "\n\n")
    if not moderation_logs:
        f.write("(No moderation interventions)\n\n")
        return
    for mod in moderation_logs:
        team_name = _safe_text(mod.get("team_name", "Unknown"))
        question = _safe_text(mod.get("question", ""))
        intervention_type = _safe_text(mod.get("intervention_type", ""))
        f.write(f"[MODERATOR → {team_name}] ({intervention_type})\n")
        f.write(f"{question}\n\n")


def _write_turns(f, turn_logs: List[Dict[str, Any]], phase: str, title: str) -> None:
    f.write("=" * 80 + "\n")
    f.write(f"PHASE: {title}\n")
    f.write("=" * 80 + "\n\n")

    phase_turns = [t for t in turn_logs if t.get("phase") == phase]
    if not phase_turns:
        f.write(f"(No {phase} turns recorded)\n\n")
        return

    for turn in phase_turns:
        turn_num = _safe_text(turn.get("turn_number", "?"))
        agent_name = _safe_text(turn.get("agent_name", "Unknown"))
        team_name = _safe_text(turn.get("team_name", "Unknown"))
        message = _safe_text(turn.get("message_content", ""))
        f.write("─" * 80 + "\n")
        f.write(f"TURN {turn_num} | {agent_name} (Team: {team_name})\n")
        f.write("─" * 80 + "\n")
        f.write(f"{message}\n\n")


def export_transcript(input_path: Path, output_path: Path) -> Path:
    data = _load_json(input_path)
    session_info = data.get("session_info", {})
    turn_logs = data.get("turn_logs", [])
    moderation_logs = data.get("moderation_logs", [])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        _write_header(f, session_info)
        _write_turns(f, turn_logs, "deliberation", "TEAM DELIBERATION")
        _write_moderation(f, moderation_logs)
        _write_turns(f, turn_logs, "debate", "INTER-TEAM DEBATE")

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export readable transcript from debate JSON log.")
    parser.add_argument("--input", required=True, help="Path to JSON log file")
    parser.add_argument("--output", required=True, help="Path to output TXT file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input log not found: {input_path}")

    exported = export_transcript(input_path, output_path)
    print(f"Transcript saved: {exported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
