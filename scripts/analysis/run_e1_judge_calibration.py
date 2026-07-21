"""
E1 judge calibration / validation gate.

Loads a labeled argument-quality dataset and evaluates correlation
between heuristic judge scores and gold labels.

Outputs:
- analysis/judge_e1/e1_results.json
- analysis/judge_e1/e1_summary.csv
- analysis/judge_e1/status.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.analysis.offline_judge import WebisAlignedOfflineJudge


def _read_dataset(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("items", [])
        return list(data)

    if path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    raise ValueError("Unsupported dataset format. Use .json or .csv")


def _resolve_key(item: Dict[str, Any], requested: str, fallbacks: List[str]) -> str | None:
    if requested and requested in item:
        return requested
    for key in fallbacks:
        if key in item:
            return key
    return None


def _pearson(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or not x:
        return 0.0
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den_x = sum((a - mx) ** 2 for a in x)
    den_y = sum((b - my) ** 2 for b in y)
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x ** 0.5 * den_y ** 0.5)


def _mae(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or not x:
        return 1.0
    return sum(abs(a - b) for a, b in zip(x, y)) / len(x)


def _spearman(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation (rank-based Pearson, handles ties via average rank)."""
    n = len(x)
    if n != len(y) or n < 2:
        return 0.0

    def _ranks(vals: List[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and vals[order[j]] == vals[order[i]]:
                j += 1
            avg_rank = (i + j - 1) / 2.0
            for k in range(i, j):
                ranks[order[k]] = avg_rank
            i = j
        return ranks

    return _pearson(_ranks(x), _ranks(y))


def _normalize_labels(golds: List[float], mode: str) -> Tuple[List[float], str]:
    if not golds:
        return golds, "none"

    min_g = min(golds)
    max_g = max(golds)

    if mode == "auto":
        if min_g < 0.0 or max_g > 1.0:
            mode = "sigmoid"
        else:
            mode = "none"

    if mode == "none":
        return golds, "none"

    if mode == "minmax":
        if max_g == min_g:
            return [0.5 for _ in golds], "minmax"
        return [
            (g - min_g) / (max_g - min_g)
            for g in golds
        ], "minmax"

    if mode == "zscore":
        mean_g = sum(golds) / len(golds)
        var = sum((g - mean_g) ** 2 for g in golds) / len(golds)
        stdev = var ** 0.5
        if stdev == 0:
            return [0.0 for _ in golds], "zscore"
        return [(g - mean_g) / stdev for g in golds], "zscore"

    if mode == "sigmoid":
        # Map z-scored labels to (0, 1) using the normal CDF.
        return [0.5 * (1.0 + math.erf(g / (2.0 ** 0.5))) for g in golds], "sigmoid"

    raise ValueError(f"Unsupported label normalization: {mode}")


def main() -> int:
    parser = argparse.ArgumentParser(description="E1 calibration for offline judge")
    parser.add_argument("--dataset", required=True, help="Path to labeled dataset (.json or .csv)")
    parser.add_argument("--label-key", default="label", help="Field name for gold label (0-1)")
    parser.add_argument("--text-key", default="text", help="Field name for argument text")
    parser.add_argument("--context-key", default="context", help="Field name for context text")
    parser.add_argument("--min-pearson", type=float, default=0.3, help="Minimum Pearson correlation to pass (default: 0.3)")
    parser.add_argument("--min-spearman", type=float, default=0.3, help="Minimum Spearman correlation to pass (default: 0.3)")
    parser.add_argument("--max-mae", type=float, default=0.35, help="Maximum MAE to pass")
    parser.add_argument(
        "--label-normalization",
        choices=["auto", "none", "minmax", "zscore", "sigmoid"],
        default="auto",
        help="Normalize labels before evaluation (auto detects z-scored labels)",
    )
    parser.add_argument("--output-dir", default="analysis/judge_e1", help="Output directory")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    items = _read_dataset(dataset_path)

    judge = WebisAlignedOfflineJudge()
    preds: List[float] = []
    golds_raw: List[float] = []
    results: List[Dict[str, Any]] = []
    print(f"[E1] Scoring {len(items)} arguments with LLM judge ...")

    text_keys = ["text", "argument", "claim", "response", "premise"]
    context_keys = ["context", "prompt", "topic", "opponent", "source", "claim_context"]
    label_keys = ["label", "score", "quality", "combined_quality", "waq", "WAQ", "argquality"]

    for item in items:
        text_key = _resolve_key(item, args.text_key, text_keys)
        label_key = _resolve_key(item, args.label_key, label_keys)
        context_key = _resolve_key(item, args.context_key, context_keys)

        if not text_key or not label_key:
            continue

        text = str(item.get(text_key, ""))
        context = str(item.get(context_key, "")) if context_key else ""
        label_raw = item.get(label_key)
        try:
            gold = float(label_raw)
        except (TypeError, ValueError):
            continue

        score, _raw, _parse_mode, _rationale = judge.score_argument(text, context)
        preds.append(score.overall)
        golds_raw.append(gold)
        results.append({
            "text": text[:200],
            "context": context[:200],
            "gold_raw": gold,
            "pred": score.overall,
            "clarity": score.clarity,
            "cogency": score.cogency,
            "relevance": score.relevance,
        })
        if len(preds) % 10 == 0:
            print(f"  [{len(preds)}/{len(items)}] scored ...", flush=True)

    golds, label_norm = _normalize_labels(golds_raw, args.label_normalization)

    if preds and golds:
        def _stdev(vals: List[float]) -> float:
            if len(vals) < 2:
                return 0.0
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            return var ** 0.5

        print(
            f"Label stdev (raw): {_stdev(golds_raw):.3f} | "
            f"Label stdev (norm): {_stdev(golds):.3f} | "
            f"Pred stdev: {_stdev(preds):.3f} | "
            f"Label norm: {label_norm}"
        )

    for row, gold in zip(results, golds):
        row["gold"] = gold

    pearson = _pearson(preds, golds)
    spearman = _spearman(preds, golds)
    mae = _mae(preds, golds)
    passed = pearson >= args.min_pearson and spearman >= args.min_spearman and mae <= args.max_mae

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "e1_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": results,
                "metrics": {
                    "pearson": pearson,
                    "spearman": spearman,
                    "mae": mae,
                    "label_normalization": label_norm,
                    "n": len(preds),
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    summary_path = output_dir / "e1_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "n",
                "pearson",
                "spearman",
                "mae",
                "passed",
                "label_normalization",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "dataset": str(dataset_path),
            "n": len(preds),
            "pearson": pearson,
            "spearman": spearman,
            "mae": mae,
            "passed": passed,
            "label_normalization": label_norm,
        })

    status_path = output_dir / "status.json"
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump({
            "passed": passed,
            "pearson": pearson,
            "spearman": spearman,
            "mae": mae,
            "n": len(preds),
            "min_pearson": args.min_pearson,
            "min_spearman": args.min_spearman,
            "max_mae": args.max_mae,
            "dataset": str(dataset_path),
            "label_normalization": label_norm,
        }, f, indent=2, ensure_ascii=False)

    print(f"E1 calibration {'PASSED' if passed else 'FAILED'}")
    print(f"  Pearson r  : {pearson:.3f}  (min={args.min_pearson})")
    print(f"  Spearman rs: {spearman:.3f}  (min={args.min_spearman})")
    print(f"  MAE        : {mae:.3f}  (max={args.max_mae})")
    print(f"  N          : {len(preds)}")
    print(f"Artifacts saved to: {output_dir}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
