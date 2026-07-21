"""
E2 benchmark harness (post-E1) + DIS computation.

Purpose:
    - Fast, flexible runner for ad-hoc experiments and quick checks.
    - NOT the publication-ready corrected E2 batch.

Runs N debates per CFR condition with max-turns=10, scores logs with the
offline judge, computes DIS, and writes aggregate summaries.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure repo root and src/ are on sys.path for direct execution
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import yaml

from src.kg_cfr.aegis_orchestrator import TripartiteStandoff, create_default_modules, get_condition_label
from config.settings import DILEMMAS_AEGIS_FILE
from scripts.analysis.offline_judge import (
    WebisAlignedOfflineJudge,
    check_e1_passed,
    score_log_file,
)
from scripts.analysis.compute_kes_metrics import compute_kes_metrics
from scripts.analysis.compute_v5_metrics import compute_v5_metrics
from scripts.analysis.compute_dis import compute_log_dis
from scripts.experiment_resilience import initialize_experiment_run

# Silence noisy LangChain deprecation warnings to keep exit code clean.
warnings.filterwarnings("ignore", category=DeprecationWarning)


def _load_dilemma_record(
    dilemma_text: str | None,
    dilemma_id: str | None,
    dilemmas_file: Path,
) -> Dict[str, str]:
    if dilemma_text:
        return {
            "id": "custom_prompt",
            "title": "Custom prompt",
            "prompt": dilemma_text,
        }
    if not dilemmas_file.exists():
        raise FileNotFoundError(f"Dilemmas file not found: {dilemmas_file}")
    with dilemmas_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    dilemmas = data.get("dilemmas", [])
    if not dilemmas:
        raise ValueError("No dilemmas found in dilemmas.yml")
    if dilemma_id:
        for d in dilemmas:
            if d.get("id") == dilemma_id:
                return {
                    "id": str(d.get("id") or ""),
                    "title": str(d.get("title") or ""),
                    "prompt": str(d.get("prompt") or ""),
                }
        raise ValueError(f"Dilemma id not found: {dilemma_id}")
    first = dilemmas[0]
    return {
        "id": str(first.get("id") or ""),
        "title": str(first.get("title") or ""),
        "prompt": str(first.get("prompt") or ""),
    }


def _create_policy_modules() -> Dict[str, Any]:
    """Create 3 AEGIS policy modules (EOH/IVC/SHM) with hidden axiomatic corpora."""
    return create_default_modules()


def _mean_ci(values: List[float]) -> Tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    se = math.sqrt(var) / math.sqrt(len(values))
    return mean, 1.96 * se


def main() -> int:
    parser = argparse.ArgumentParser(description="E2 benchmark harness + DIS")
    parser.add_argument("--dilemma", default=None, help="Override dilemma text")
    parser.add_argument("--dilemma-id", default=None, help="Dilemma id from dilemmas_aegis.yml")
    parser.add_argument("--dilemmas-file", default=str(DILEMMAS_AEGIS_FILE), help="Path to AEGIS dilemmas file")
    parser.add_argument("--runs", type=int, default=50, help="Runs per condition (default: 50)")
    parser.add_argument("--max-turns", type=int, default=10, help="Debate turns (default: 10)")
    parser.add_argument(
        "--conditions",
        default="no_cfr_baseline,cfr_no_kg,kg_cfr_full",
        help=(
            "Comma-separated CFR condition names. "
            "Canonical: no_cfr_baseline, cfr_no_kg, kg_cfr_full, cfr_no_kg_no_idrag. "
            "Legacy aliases also accepted: none, screen_no_idrag_no_cfr, prompt-only, knowledge-grounded."
        ),
    )
    parser.add_argument("--skip-e1", action="store_true", help="Skip E1 gate (not recommended)")
    parser.add_argument("--e1-status", default="analysis/judge_e1/status.json")
    parser.add_argument("--output-dir", default="analysis/outputs/e2")
    parser.add_argument("--no-shocks", action="store_true", help="Disable SystemDispatcher shocks")
    parser.add_argument("--shock-type", default="auto", help="Shock type: S1|S2|S3|auto")
    parser.add_argument("--shock-seed", type=int, default=42, help="Base seed for shocks")
    parser.add_argument("--no-tom", action="store_true", help="Disable ToM-Lite")
    parser.add_argument("--no-idrag", action="store_true", help="Disable ID-RAG")
    parser.add_argument("--no-adaptive-idrag", action="store_true", help="Disable adaptive ID-RAG")
    args = parser.parse_args()

    require_e1 = not args.skip_e1
    enable_shocks = not args.no_shocks
    use_tom = not args.no_tom
    use_idrag = not args.no_idrag
    use_adaptive_idrag = not args.no_adaptive_idrag

    if require_e1:
        ok, reason = check_e1_passed(Path(args.e1_status))
        if not ok:
            print(f"E1 gate failed: {reason}")
            return 2

    dilemma = _load_dilemma_record(args.dilemma, args.dilemma_id, Path(args.dilemmas_file))
    dilemma_prompt = dilemma.get("prompt", "")
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    progress_path = Path("logs/experiments/progress/e2_aegis_benchmark.jsonl")
    logger, run_id, completed = initialize_experiment_run(
        progress_path,
        "exp_e2",
        metadata={
            "runs_per_condition": args.runs,
            "max_turns": args.max_turns,
            "conditions": conditions,
            "shock_type": args.shock_type if enable_shocks else "none",
        },
    )

    run_index = 0
    log_paths: List[Path] = []
    condition_map: Dict[str, str] = {}  # session_id → canonical cfr_mode
    label_map: Dict[str, str] = {}     # session_id → 6-cell condition_label

    for condition in conditions:
        for i in range(args.runs):
            exp_id = f"{run_id}_{condition}_{i:03d}"
            if exp_id in completed:
                log_path = Path("logs/experiments") / f"{exp_id}.json"
                if log_path.exists():
                    log_paths.append(log_path)
                    condition_map[exp_id] = condition
                    label_map[exp_id] = get_condition_label(condition, use_idrag, use_adaptive_idrag)
                continue

            session_id = exp_id  # ensures logs/experiments
            print(f"\n[E2] Running: {session_id}")

            modules = _create_policy_modules()
            standoff = TripartiteStandoff(
                debate_turns=args.max_turns,
                enable_shocks=enable_shocks,
                shock_type=args.shock_type if enable_shocks else "none",
                shock_seed=args.shock_seed + run_index,
                cfr_mode=condition,
                dilemma_id=dilemma.get("id"),
                session_id=session_id,
                enable_logging=True,
            )

            try:
                standoff.start_standoff(
                    dilemma_prompt,
                    modules,
                    use_id_rag=use_idrag,
                    use_adaptive_idrag=use_adaptive_idrag,
                )

                log_path = Path("logs/experiments") / f"{session_id}.json"
                if log_path.exists():
                    log_paths.append(log_path)
                    condition_map[session_id] = condition
                    label_map[session_id] = get_condition_label(condition, use_idrag, use_adaptive_idrag)
                    logger.log_experiment_result(
                        run_id,
                        {
                            "experiment_id": exp_id,
                            "status": "success",
                            "condition": condition,
                            "json_log": str(log_path),
                        },
                    )
                else:
                    logger.log_experiment_result(
                        run_id,
                        {
                            "experiment_id": exp_id,
                            "status": "failed",
                            "condition": condition,
                            "error": "missing_log",
                        },
                    )
            except Exception as exc:
                logger.log_experiment_result(
                    run_id,
                    {
                        "experiment_id": exp_id,
                        "status": "failed",
                        "condition": condition,
                        "error": str(exc),
                    },
                )

            # Best-effort GPU/CPU cleanup between runs to avoid embedding model OOM.
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            run_index += 1

    if not log_paths:
        print("No logs produced; aborting scoring.")
        return 1

    output_dir = Path(args.output_dir) / run_id
    judge_dir = output_dir / "judge"
    dis_dir = output_dir / "dis"
    judge_dir.mkdir(parents=True, exist_ok=True)
    dis_dir.mkdir(parents=True, exist_ok=True)

    # --- Judge scoring ---
    judge = WebisAlignedOfflineJudge()
    judge_rows: List[Dict[str, Any]] = []
    judge_summaries: List[Dict[str, Any]] = []
    judge_results: Dict[str, Dict[str, Any]] = {}

    for log_path in log_paths:
        session_id = log_path.stem
        out_json = judge_dir / f"judge_{session_id}.json"
        if out_json.exists():
            result = json.loads(out_json.read_text(encoding="utf-8"))
        else:
            result = score_log_file(log_path, judge)
            session_id = result.get("session_id") or session_id
            out_json = judge_dir / f"judge_{session_id}.json"
            with out_json.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        session_id = result.get("session_id") or session_id

        judge_results[session_id] = result

        summary = dict(result["summary"])
        summary["session_id"] = session_id
        summary["condition"] = condition_map.get(session_id, "unknown")
        summary["condition_label"] = label_map.get(session_id, summary["condition"])
        judge_summaries.append(summary)

        for row in result["per_turn"]:
            row = dict(row)
            row["session_id"] = session_id
            row["condition"] = condition_map.get(session_id, "unknown")
            row["condition_label"] = label_map.get(session_id, row["condition"])
            judge_rows.append(row)

    # --- DIS scoring ---
    dis_rows: List[Dict[str, Any]] = []
    dis_summaries: List[Dict[str, Any]] = []

    for idx, log_path in enumerate(log_paths, start=1):
        print(f"[dis] {idx}/{len(log_paths)}: {log_path.stem}")
        session_id = log_path.stem
        out_json = dis_dir / f"dis_{session_id}.json"
        if out_json.exists():
            result = json.loads(out_json.read_text(encoding="utf-8"))
        else:
            result = compute_log_dis(log_path, mode="jaccard")
            session_id = result.get("session_id") or session_id
            out_json = dis_dir / f"dis_{session_id}.json"
            with out_json.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        summary = dict(result["summary"])
        summary["session_id"] = session_id
        summary["condition"] = condition_map.get(session_id, "unknown")
        summary["condition_label"] = label_map.get(session_id, summary["condition"])
        dis_summaries.append(summary)

        for row in result["per_turn"]:
            row = dict(row)
            row["session_id"] = session_id
            row["condition"] = condition_map.get(session_id, "unknown")
            row["condition_label"] = label_map.get(session_id, row["condition"])
            dis_rows.append(row)

    # --- KES metrics ---
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    kes_rows: List[Dict[str, Any]] = []
    kes_summaries: List[Dict[str, Any]] = []

    for idx, log_path in enumerate(log_paths, start=1):
        print(f"[kes] {idx}/{len(log_paths)}: {log_path.stem}")
        session_id = log_path.stem
        out_json = metrics_dir / f"kes_{session_id}.json"
        if out_json.exists():
            result = json.loads(out_json.read_text(encoding="utf-8"))
        else:
            judge_result = judge_results.get(session_id)
            result = compute_kes_metrics(log_path, judge_result=judge_result)
            session_id = result.get("session_id") or session_id
            out_json = metrics_dir / f"kes_{session_id}.json"
            with out_json.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        summary = dict(result["summary"])
        summary["session_id"] = session_id
        summary["condition"] = condition_map.get(session_id, "unknown")
        summary["condition_label"] = label_map.get(session_id, summary["condition"])
        kes_summaries.append(summary)

        for row in result["per_turn"]:
            row = dict(row)
            row["session_id"] = session_id
            row["condition"] = condition_map.get(session_id, "unknown")
            row["condition_label"] = label_map.get(session_id, row["condition"])
            kes_rows.append(row)

    # --- v5 deterministic metrics ---
    v5_rows: List[Dict[str, Any]] = []
    v5_summaries: List[Dict[str, Any]] = []

    for idx, log_path in enumerate(log_paths, start=1):
        print(f"[v5] {idx}/{len(log_paths)}: {log_path.stem}")
        session_id = log_path.stem
        out_json = metrics_dir / f"v5_{session_id}.json"
        if out_json.exists():
            result = json.loads(out_json.read_text(encoding="utf-8"))
        else:
            result = compute_v5_metrics(log_path)
            session_id = result.get("session_id") or session_id
            out_json = metrics_dir / f"v5_{session_id}.json"
            with out_json.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        summary = dict(result.get("summary", {}))
        summary["session_id"] = session_id
        summary["condition"] = condition_map.get(session_id, "unknown")
        summary["condition_label"] = label_map.get(session_id, summary["condition"])
        v5_summaries.append(summary)

        for row in result.get("per_turn", []):
            row = dict(row)
            row["session_id"] = session_id
            row["condition"] = condition_map.get(session_id, "unknown")
            row["condition_label"] = label_map.get(session_id, row["condition"])
            v5_rows.append(row)

    # --- Aggregate summary ---
    aggregate: List[Dict[str, Any]] = []
    for condition in conditions:
        judge_vals = [
            s["avg_overall"]
            for s in judge_summaries
            if s.get("condition") == condition and s.get("avg_overall") is not None
        ]
        dis_vals = [
            s["avg_dis"]
            for s in dis_summaries
            if s.get("condition") == condition and s.get("avg_dis") is not None
        ]
        cc_vals = [
            s["avg_cc"]
            for s in kes_summaries
            if s.get("condition") == condition and s.get("avg_cc") is not None
        ]
        prr_vals = [
            s["avg_prr"]
            for s in kes_summaries
            if s.get("condition") == condition and s.get("avg_prr") is not None
        ]
        aca_vals = [
            s["avg_aca"]
            for s in kes_summaries
            if s.get("condition") == condition and s.get("avg_aca") is not None
        ]
        v5_cc_vals = [
            s.get("avg_cc")
            for s in v5_summaries
            if s.get("condition") == condition and s.get("avg_cc") is not None
        ]
        v5_dis_vals = [
            s.get("avg_dis")
            for s in v5_summaries
            if s.get("condition") == condition and s.get("avg_dis") is not None
        ]
        v5_sr_vals = [
            s.get("avg_sr")
            for s in v5_summaries
            if s.get("condition") == condition and s.get("avg_sr") is not None
        ]

        judge_mean, judge_ci = _mean_ci(judge_vals)
        dis_mean, dis_ci = _mean_ci(dis_vals)
        cc_mean, cc_ci = _mean_ci(cc_vals)
        prr_mean, prr_ci = _mean_ci(prr_vals)
        aca_mean, aca_ci = _mean_ci(aca_vals)
        v5_cc_mean, v5_cc_ci = _mean_ci(v5_cc_vals)
        v5_dis_mean, v5_dis_ci = _mean_ci(v5_dis_vals)
        v5_sr_mean, v5_sr_ci = _mean_ci(v5_sr_vals)

        aggregate.append({
            "condition": condition,
            "condition_label": get_condition_label(condition, use_idrag, use_adaptive_idrag),
            "judge_avg_overall_mean": judge_mean,
            "judge_avg_overall_ci95": judge_ci,
            "dis_mean": dis_mean,
            "dis_ci95": dis_ci,
            "cc_mean": cc_mean,
            "cc_ci95": cc_ci,
            "prr_mean": prr_mean,
            "prr_ci95": prr_ci,
            "aca_mean": aca_mean,
            "aca_ci95": aca_ci,
            "v5_cc_mean": v5_cc_mean,
            "v5_cc_ci95": v5_cc_ci,
            "v5_dis_mean": v5_dis_mean,
            "v5_dis_ci95": v5_dis_ci,
            "v5_sr_mean": v5_sr_mean,
            "v5_sr_ci95": v5_sr_ci,
            "n_judge": len(judge_vals),
            "n_dis": len(dis_vals),
            "n_cc": len(cc_vals),
            "n_prr": len(prr_vals),
            "n_aca": len(aca_vals),
            "n_v5_cc": len(v5_cc_vals),
            "n_v5_dis": len(v5_dis_vals),
            "n_v5_sr": len(v5_sr_vals),
        })

    # Save outputs
    (output_dir / "summary").mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary" / "e2_aggregate.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)

    def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            import csv
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    _write_csv(judge_dir / "judge_scores_turns.csv", judge_rows)
    _write_csv(judge_dir / "judge_scores_summary.csv", judge_summaries)
    _write_csv(dis_dir / "dis_scores_turns.csv", dis_rows)
    _write_csv(dis_dir / "dis_scores_summary.csv", dis_summaries)
    _write_csv(metrics_dir / "kes_metrics_turns.csv", kes_rows)
    _write_csv(metrics_dir / "kes_metrics_summary.csv", kes_summaries)
    _write_csv(metrics_dir / "v5_metrics_turns.csv", v5_rows)
    _write_csv(metrics_dir / "v5_metrics_summary.csv", v5_summaries)
    _write_csv(output_dir / "summary" / "e2_aggregate.csv", aggregate)

    logger.log_run_complete(run_id, summary={"logs_scored": len(log_paths)})
    print(f"E2 artifacts saved under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
