import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "Turn_Context_ID",
    "Condition",
    "Response_Text",
    "Judge_Overall_Score",
}

JUDGE_REQUIRED_COLUMNS = {
    "turn_number",
    "module_id",
    "overall",
    "session_id",
    "condition",
}


def _collect_csv_files(input_dirs: List[str], pattern: str) -> List[Path]:
    files: List[Path] = []
    for dir_path in input_dirs:
        dir_path = str(dir_path).strip()
        if not dir_path:
            continue
        for csv_path in glob.glob(str(Path(dir_path) / pattern)):
            files.append(Path(csv_path))
    return files


def _load_logs(csv_files: List[Path]) -> pd.DataFrame:
    if not csv_files:
        raise FileNotFoundError("No CSV files found for the provided inputs.")

    frames = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        missing = REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in {csv_path}: {sorted(missing)}"
            )
        frames.append(df[list(REQUIRED_COLUMNS)])

    return pd.concat(frames, ignore_index=True)


def _load_session_log(log_path: Path) -> Dict[Tuple[int, str], Dict[str, str]]:
    with log_path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)

    dilemma_id = payload.get("session_info", {}).get("metadata", {}).get("dilemma_id")
    if not dilemma_id:
        dilemma_id = "unknown"

    turn_map: Dict[Tuple[int, str], Dict[str, str]] = {}
    for entry in payload.get("turn_logs", []):
        turn_number = entry.get("turn_number")
        module_id = entry.get("module_id")
        message_content = entry.get("message_content")
        if turn_number is None or module_id is None or message_content is None:
            continue
        turn_map[(int(turn_number), str(module_id))] = {
            "message_content": str(message_content),
            "dilemma_id": str(dilemma_id),
        }

    return turn_map


def _load_turns_from_experiments(input_dirs: List[str]) -> pd.DataFrame:
    rows = []
    for dir_path in input_dirs:
        exp_dir = Path(dir_path)
        judge_csv = exp_dir / "judge" / "judge_scores_turns.csv"
        if not judge_csv.exists():
            continue

        judge_df = pd.read_csv(judge_csv)
        missing = JUDGE_REQUIRED_COLUMNS.difference(judge_df.columns)
        if missing:
            raise ValueError(
                f"Missing required judge columns in {judge_csv}: {sorted(missing)}"
            )

        for session_id, session_df in judge_df.groupby("session_id", sort=False):
            log_path = Path("logs") / "experiments" / f"{session_id}.json"
            if not log_path.exists():
                continue

            turn_map = _load_session_log(log_path)
            run_id = str(session_id).split("_")[-1]

            for _, row in session_df.iterrows():
                key = (int(row["turn_number"]), str(row["module_id"]))
                if key not in turn_map:
                    continue
                turn_meta = turn_map[key]
                turn_context_id = (
                    f"{turn_meta['dilemma_id']}:{run_id}:{key[1]}:{key[0]}"
                )
                rows.append(
                    {
                        "Turn_Context_ID": turn_context_id,
                        "Condition": row["condition"],
                        "Response_Text": turn_meta["message_content"],
                        "Judge_Overall_Score": float(row["overall"]),
                    }
                )

    if not rows:
        raise FileNotFoundError(
            "No usable judge scores or logs found in the provided experiment dirs."
        )

    return pd.DataFrame(rows)


def _filter_conditions(df: pd.DataFrame, conditions: List[str]) -> pd.DataFrame:
    if not conditions:
        return df
    return df[df["Condition"].isin(conditions)].copy()


def _build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    pairs = []
    grouped = df.groupby("Turn_Context_ID", sort=False)

    for turn_id, group in grouped:
        group = group.dropna(subset=["Condition", "Response_Text", "Judge_Overall_Score"])
        if group["Condition"].nunique() != 2:
            # Skip contexts that do not have exactly two conditions after filtering.
            continue
        if len(group) != 2:
            # Ensure exactly two rows for the pair.
            group = group.drop_duplicates(subset=["Condition"]).head(2)
        if len(group) != 2:
            continue

        row_a, row_b = group.iloc[0], group.iloc[1]
        score_a = float(row_a["Judge_Overall_Score"])
        score_b = float(row_b["Judge_Overall_Score"])
        pairs.append(
            {
                "Turn_Context_ID": turn_id,
                "Condition_A": row_a["Condition"],
                "Condition_B": row_b["Condition"],
                "Text_A": row_a["Response_Text"],
                "Text_B": row_b["Response_Text"],
                "Judge_Score_A": score_a,
                "Judge_Score_B": score_b,
                "Score_Delta": abs(score_a - score_b),
            }
        )

    return pd.DataFrame(pairs)


def _bucketize(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    high = df[df["Score_Delta"] > 0.20].copy()
    medium = df[(df["Score_Delta"] > 0.05) & (df["Score_Delta"] <= 0.20)].copy()
    low = df[df["Score_Delta"] <= 0.05].copy()
    return high, medium, low


def _sample_bucket(
    rng: np.random.Generator,
    bucket: pd.DataFrame,
    count: int,
    replace: bool = False,
) -> pd.DataFrame:
    if bucket.empty or count <= 0:
        return bucket.iloc[0:0].copy()
    return bucket.sample(n=count, replace=replace, random_state=rng.integers(0, 2**32 - 1))


def _stratified_sample(
    df: pd.DataFrame,
    rng: np.random.Generator,
    target_high: int = 30,
    target_medium: int = 40,
    target_low: int = 30,
) -> pd.DataFrame:
    high, medium, low = _bucketize(df)

    sample_high = _sample_bucket(rng, high, min(target_high, len(high)))
    sample_low = _sample_bucket(rng, low, min(target_low, len(low)))

    remaining_high = target_high - len(sample_high)
    remaining_low = target_low - len(sample_low)
    extra_needed = max(0, remaining_high + remaining_low)

    sample_medium = _sample_bucket(rng, medium, min(target_medium, len(medium)))

    # If high/low are short, oversample medium to hit the total 100.
    if extra_needed > 0:
        medium_remaining = medium.drop(sample_medium.index, errors="ignore")
        if len(medium_remaining) >= extra_needed:
            extra_medium = _sample_bucket(rng, medium_remaining, extra_needed)
        else:
            # Fall back to sampling with replacement from medium.
            extra_medium = _sample_bucket(rng, medium, extra_needed, replace=True)
        sample_medium = pd.concat([sample_medium, extra_medium], ignore_index=True)

    combined = pd.concat([sample_high, sample_medium, sample_low], ignore_index=True)

    # If still short (e.g., insufficient data overall), pad by sampling medium with replacement.
    if len(combined) < 100 and not medium.empty:
        deficit = 100 - len(combined)
        pad = _sample_bucket(rng, medium, deficit, replace=True)
        combined = pd.concat([combined, pad], ignore_index=True)

    return combined.sample(frac=1.0, random_state=rng.integers(0, 2**32 - 1)).reset_index(drop=True)


def _assign_blind_labels(df: pd.DataFrame, rng: np.random.Generator) -> Tuple[pd.DataFrame, pd.DataFrame]:
    blind_rows = []
    key_rows = []

    for idx, row in df.iterrows():
        pair_id = f"pair_{idx + 1:04d}"
        swap = rng.random() < 0.5

        if swap:
            text_a, text_b = row["Text_B"], row["Text_A"]
            cond_a, cond_b = row["Condition_B"], row["Condition_A"]
            score_a, score_b = row["Judge_Score_B"], row["Judge_Score_A"]
        else:
            text_a, text_b = row["Text_A"], row["Text_B"]
            cond_a, cond_b = row["Condition_A"], row["Condition_B"]
            score_a, score_b = row["Judge_Score_A"], row["Judge_Score_B"]

        blind_rows.append(
            {
                "Pair_ID": pair_id,
                "Text_A": text_a,
                "Text_B": text_b,
            }
        )
        key_rows.append(
            {
                "Pair_ID": pair_id,
                "Condition_A": cond_a,
                "Condition_B": cond_b,
                "Judge_Score_A": score_a,
                "Judge_Score_B": score_b,
                "Score_Delta": row["Score_Delta"],
            }
        )

    blind_df = pd.DataFrame(blind_rows)
    key_df = pd.DataFrame(key_rows)
    return blind_df, key_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a stratified blind spot-check sample for human evaluation."
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        required=True,
        help="One or more directories containing CSV logs.",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern for CSV files in each input directory (default: *.csv).",
    )
    parser.add_argument(
        "--conditions",
        default="",
        help="Comma-separated conditions to keep (default: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for reproducibility (default: 1337).",
    )
    parser.add_argument(
        "--blind-out",
        default="human_spotcheck_blind.csv",
        help="Output CSV path for blind evaluation file.",
    )
    parser.add_argument(
        "--key-out",
        default="human_spotcheck_key.csv",
        help="Output CSV path for answer key file.",
    )

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    csv_files = _collect_csv_files(args.input_dirs, args.pattern)
    if csv_files:
        df = _load_logs(csv_files)
    else:
        df = _load_turns_from_experiments(args.input_dirs)
    df = _filter_conditions(df, conditions)

    pairs_df = _build_pairs(df)
    if pairs_df.empty:
        raise ValueError("No valid pairs found after filtering. Check inputs/conditions.")

    sampled = _stratified_sample(pairs_df, rng)
    blind_df, key_df = _assign_blind_labels(sampled, rng)

    blind_df.to_csv(args.blind_out, index=False)
    key_df.to_csv(args.key_out, index=False)


if __name__ == "__main__":
    main()
