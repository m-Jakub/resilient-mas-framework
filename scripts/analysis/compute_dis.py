"""
Compute DIS (Dialectical Interactivity Score) from debate logs.

DIS is defined as the per-turn overlap between opponent concepts and the
current response concepts, derived from logged `concepts_mentioned`.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_log_paths(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.json") if p.is_file())


def _write_csv(output_csv: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _find_last_opponent_turn(
    debate_turns: List[Dict[str, Any]], idx: int, module_id: str
) -> Optional[Dict[str, Any]]:
    for j in range(idx - 1, -1, -1):
        prev_id = debate_turns[j].get("module_id") or debate_turns[j].get("team_id")
        if prev_id != module_id:
            return debate_turns[j]
    return None


def _concept_set(turn: Dict[str, Any]) -> set[str]:
    concepts = turn.get("concepts_mentioned") or []
    if isinstance(concepts, str):
        return {c.strip() for c in concepts.split(",") if c.strip()}
    return {str(c).strip() for c in concepts if str(c).strip()}


def _compute_overlap(a: set[str], b: set[str], mode: str) -> float:
    if not a and not b:
        return 0.0
    if mode == "opponent_recall":
        return len(a & b) / max(1, len(a))
    # Default: Jaccard
    union = a | b
    return len(a & b) / max(1, len(union))


def compute_log_dis(log_path: Path, mode: str) -> Dict[str, Any]:
    with log_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    turn_logs = data.get("turn_logs", [])
    # Logs use phase="standoff" for the main debate phase; accept both names.
    debate_turns = [t for t in turn_logs if t.get("phase") in {"debate", "standoff"}]

    per_turn: List[Dict[str, Any]] = []
    for idx, turn in enumerate(debate_turns):
        module_id = turn.get("module_id") or turn.get("team_id", "")
        opponent_turn = _find_last_opponent_turn(debate_turns, idx, module_id)
        if not opponent_turn:
            continue

        current_concepts = _concept_set(turn)
        opponent_concepts = _concept_set(opponent_turn)
        dis_score = _compute_overlap(opponent_concepts, current_concepts, mode)

        per_turn.append({
            "turn_number": turn.get("turn_number"),
            "module_id": module_id,
            "module_label": turn.get("module_label"),
            "opponent_module_id": opponent_turn.get("module_id") or opponent_turn.get("team_id"),
            "dis_score": dis_score,
            "intersection_size": len(opponent_concepts & current_concepts),
            "opponent_concepts": len(opponent_concepts),
            "current_concepts": len(current_concepts),
        })

    if per_turn:
        avg_dis = sum(t["dis_score"] for t in per_turn) / len(per_turn)
    else:
        avg_dis = None

    return {
        "log_path": str(log_path),
        "session_id": data.get("session_info", {}).get("session_id"),
        "summary": {
            "turns_scored": len(per_turn),
            "avg_dis": avg_dis,
            "mode": mode,
        },
        "per_turn": per_turn,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute DIS over debate logs")
    parser.add_argument("--logs", required=True, help="Path to a log JSON file or a directory")
    parser.add_argument("--output-dir", default="analysis/outputs/dis", help="Output directory")
    parser.add_argument(
        "--mode",
        choices=["jaccard", "opponent_recall"],
        default="jaccard",
        help="Overlap mode (default: jaccard)",
    )
    args = parser.parse_args()

    logs_path = Path(args.logs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for log_path in _load_log_paths(logs_path):
        result = compute_log_dis(log_path, args.mode)
        session_id = result.get("session_id") or log_path.stem

        out_json = output_dir / f"dis_{session_id}.json"
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        for row in result["per_turn"]:
            row = dict(row)
            row["session_id"] = session_id
            all_rows.append(row)

        summary = dict(result["summary"])
        summary["session_id"] = session_id
        summaries.append(summary)

    _write_csv(output_dir / "dis_scores_turns.csv", all_rows)
    _write_csv(output_dir / "dis_scores_summary.csv", summaries)

    print(f"Saved DIS outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
