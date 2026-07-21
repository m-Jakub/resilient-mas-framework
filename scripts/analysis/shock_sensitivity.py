import argparse
import json
import sys
from pathlib import Path
import pandas as pd
from typing import Dict, Optional


DEFAULT_CSV = "analysis/outputs/final_run_ablation_01052026/ablation_validator_iq_run_b_gemini_3_1_pro_preview.csv"
DEFAULT_JSON_DIR = Path("logs/archive/article2")

def build_json_index(json_dir: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    if not json_dir.exists():
        return index
    for p in json_dir.rglob("*.json"):
        stem = p.stem
        # prefer first seen; if collisions happen, later wins (could be changed)
        index[stem] = p
    return index


def get_shock_level(session_id: str, index: Dict[str, Path]) -> str:
    """Determine shock level for a session using indexed JSON files.

    Returns one of: 'High Shock', 'Low Shock', 'Unknown'
    """
    if not session_id:
        return "Unknown"
    path = index.get(str(session_id))
    if path is None or not path.exists():
        return "Unknown"

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return "Unknown"

    # The logs can have different shapes; be defensive
    shock_logs = data.get("shock_logs") or data.get("shocks") or []
    if not isinstance(shock_logs, list):
        return "Unknown"

    for entry in shock_logs:
        if not isinstance(entry, dict):
            continue
        meta = entry.get("metadata") or entry.get("meta") or entry
        # check multiple possible keys/values
        category = None
        severity = None
        if isinstance(meta, dict):
            category = meta.get("category") or meta.get("type") or meta.get("shock_type")
            severity = meta.get("severity")
        # normalize
        if isinstance(category, str) and category.lower() == "zero_sum":
            return "High Shock"
        if isinstance(severity, str) and severity.lower() == "high":
            return "High Shock"

    return "Low Shock"


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    # ensure columns exist
    if "initial_pf" not in df.columns:
        df["initial_pf"] = pd.NA
    if "final_pf" not in df.columns:
        df["final_pf"] = pd.NA
    if "was_divergence_report" not in df.columns:
        df["was_divergence_report"] = 0

    summary = df.groupby("shock_level").agg(
        count=("session_id", "count"),
        avg_initial_pf=("initial_pf", "mean"),
        avg_final_pf=("final_pf", "mean"),
        divergence_count=("was_divergence_report", "sum"),
    ).reset_index()
    summary["divergence_rate_%"] = (summary["divergence_count"] / summary["count"] * 100).round(1)
    summary["avg_initial_pf"] = summary["avg_initial_pf"].round(3)
    summary["avg_final_pf"] = summary["avg_final_pf"].round(3)
    return summary


def main(argv: Optional[list] = None):
    parser = argparse.ArgumentParser(description="Shock sensitivity analysis")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to results CSV")
    parser.add_argument("--json-dir", default=str(DEFAULT_JSON_DIR), help="Directory with archived JSON logs")
    parser.add_argument("--output", default=None, help="Optional path to save the summary CSV")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    json_dir = Path(args.json_dir)

    print(f"Loading CSV: {csv_path}")
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(2)

    # read csv (be permissive with encoding)
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        sys.exit(2)

    # find name for session id column
    if "session_id" not in df.columns:
        for candidate in ("session", "id", "sessionId"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "session_id"})
                break
    if "session_id" not in df.columns:
        print("ERROR: Could not find a session identifier column in CSV (expected 'session_id')")
        sys.exit(2)

    print(f"Indexing JSON files under: {json_dir} ...")
    index = build_json_index(json_dir)
    print(f"Found {len(index)} JSON files in archive")

    print("Annotating shock levels (this may take a while)...")
    df["session_id"] = df["session_id"].astype(str)
    df["shock_level"] = df["session_id"].apply(lambda s: get_shock_level(s, index))

    # drop unknowns but report how many
    unknown_count = (df["shock_level"] == "Unknown").sum()
    if unknown_count:
        print(f"Warning: {unknown_count} sessions had no matching JSON and will be ignored")
    df = df[df["shock_level"] != "Unknown"].copy()

    summary = summarize(df)

    print("\n" + "=" * 80)
    print("EXPERIMENT E1: SHOCK SENSITIVITY RESULTS")
    print("=" * 80)
    print(summary.to_string(index=False))
    print("=" * 80)

    if args.output:
        outp = Path(args.output)
        summary.to_csv(outp, index=False)
        print(f"Saved summary to: {outp}")


if __name__ == "__main__":
    main()