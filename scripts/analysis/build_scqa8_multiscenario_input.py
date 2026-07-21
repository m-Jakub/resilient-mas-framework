"""
Build a combined E2_Turns_judge_fixed.csv from multiple E2 output folders.

Inputs per folder:
- judge/judge_scores_turns.csv
- dis/dis_scores_turns.csv
- metrics/kes_metrics_turns.csv
- metrics/v5_metrics_turns.csv

Also enriches rows with:
- scenario_id (from logs/experiments/<session_id>.json -> session_info.metadata.dilemma_id)
- shock_type_global (session_info.metadata.shock_type)
- shock_type_turn (from shock_logs by turn_number)
- message_length (from turn_logs)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = REPO_ROOT / "logs" / "experiments"

COLUMN_ORDER = [
    "condition",
    "condition_label",
    "session_id",
    "debate_id",
    "scenario_id",
    "turn_number",
    "module_id",
    "module_label",
    "agent_id",
    "shock_type_global",
    "shock_type_turn",
    "message_length",
    "clarity",
    "cogency",
    "relevance",
    "overall",
    "dis_score",
    "aca",
    "prr",
    "prr_drop",
    "prr_rebound_1",
    "prr_rebound_2",
    "cc_kes",
    "cc_v5",
    "dis_v5",
    "sr_v5",
    "parse_mode",
    "has_subscores",
]


def _load_log_metadata(session_id: str) -> Dict[str, object]:
    log_path = LOGS_DIR / f"{session_id}.json"
    if not log_path.exists():
        return {
            "scenario_id": "unknown",
            "shock_type_global": "unknown",
            "shock_by_turn": {},
            "message_length": {},
        }

    payload = json.loads(log_path.read_text(encoding="utf-8"))
    session_info = payload.get("session_info", {})
    metadata = session_info.get("metadata", {})
    scenario_id = metadata.get("dilemma_id", "unknown")
    shock_type_global = metadata.get("shock_type", "unknown")

    shock_by_turn = {}
    for shock in payload.get("shock_logs", []):
        t = shock.get("turn_number")
        if t is not None:
            shock_by_turn[int(t)] = shock.get("shock_type")

    message_length = {}
    for turn in payload.get("turn_logs", []):
        t = turn.get("turn_number")
        mid = turn.get("module_id")
        if t is None or not mid:
            continue
        message_length[(int(t), str(mid))] = turn.get("message_length")

    return {
        "scenario_id": scenario_id,
        "shock_type_global": shock_type_global,
        "shock_by_turn": shock_by_turn,
        "message_length": message_length,
    }


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def _load_folder(folder: Path) -> pd.DataFrame:
    judge_csv = folder / "judge" / "judge_scores_turns.csv"
    dis_csv = folder / "dis" / "dis_scores_turns.csv"
    kes_csv = folder / "metrics" / "kes_metrics_turns.csv"
    v5_csv = folder / "metrics" / "v5_metrics_turns.csv"

    df_j = _load_csv(judge_csv)
    df_dis = _load_csv(dis_csv)
    df_kes = _load_csv(kes_csv)
    df_v5 = _load_csv(v5_csv)

    # Normalize types for merge keys.
    for df in (df_j, df_dis, df_kes, df_v5):
        df["turn_number"] = df["turn_number"].astype(int)
        df["module_id"] = df["module_id"].astype(str)
        df["session_id"] = df["session_id"].astype(str)

    df = df_j.copy()
    df["debate_id"] = df["session_id"]

    df = df.merge(
        df_dis[["session_id", "turn_number", "module_id", "dis_score"]],
        on=["session_id", "turn_number", "module_id"],
        how="left",
    )
    df = df.merge(
        df_kes[[
            "session_id",
            "turn_number",
            "module_id",
            "cc",
            "aca",
            "prr",
            "prr_drop",
            "prr_rebound_1",
            "prr_rebound_2",
        ]],
        on=["session_id", "turn_number", "module_id"],
        how="left",
    )
    df = df.merge(
        df_v5[["session_id", "turn_number", "module_id", "cc", "dis", "sr"]],
        on=["session_id", "turn_number", "module_id"],
        how="left",
        suffixes=("", "_v5"),
    )

    df = df.rename(
        columns={
            "cc": "cc_kes",
            "cc_v5": "cc_v5",
            "dis": "dis_v5",
            "sr": "sr_v5",
        }
    )

    # Enrich with scenario/shock/message_length from logs.
    scenarios = {}
    shock_types = {}
    shock_by_turn = {}
    msg_lengths = {}

    for session_id in df["session_id"].unique():
        meta = _load_log_metadata(session_id)
        scenarios[session_id] = meta["scenario_id"]
        shock_types[session_id] = meta["shock_type_global"]
        shock_by_turn[session_id] = meta["shock_by_turn"]
        msg_lengths[session_id] = meta["message_length"]

    df["scenario_id"] = df["session_id"].map(scenarios)
    df["shock_type_global"] = df["session_id"].map(shock_types)

    def _shock_turn(row):
        return shock_by_turn.get(row["session_id"], {}).get(row["turn_number"], "NA")

    def _msg_len(row):
        key = (row["turn_number"], row["module_id"])
        return msg_lengths.get(row["session_id"], {}).get(key)

    df["shock_type_turn"] = df.apply(_shock_turn, axis=1)
    df["message_length"] = df.apply(_msg_len, axis=1)

    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Build combined SCQA8 input CSV")
    parser.add_argument(
        "--folders",
        required=True,
        help="Comma-separated analysis/outputs/e2/<run_id> folders",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path for E2_Turns_judge_fixed.csv",
    )
    args = parser.parse_args()

    folders = [Path(p.strip()) for p in args.folders.split(",") if p.strip()]
    if not folders:
        raise SystemExit("No folders provided.")

    all_frames: List[pd.DataFrame] = []
    for folder in folders:
        if not folder.exists():
            raise SystemExit(f"Missing folder: {folder}")
        df = _load_folder(folder)
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.reindex(columns=COLUMN_ORDER)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    debates = combined["debate_id"].nunique()
    scenarios = combined["scenario_id"].nunique()
    print(f"Saved combined CSV: {out_path}")
    print(f"Rows: {len(combined)} | Debates: {debates} | Scenarios: {scenarios}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
