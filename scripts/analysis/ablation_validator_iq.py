"""
Ablation Study: Validator IQ Experiment

Compare Phase 4 performance with different Validator LLMs:
- Run A: Proposer=Flash, Validator=Flash (baseline)
- Run B: Proposer=Flash, Validator=Pro (expert validator)

Runs both experiments on the same sample set and generates comparative analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from src.common.api_abstraction import get_philosopher_api
from src.apg.phase4_synthesis import (
    MAX_SYNTHESIS_ITER,
    SynthesisState,
    build_synthesis_subgraph,
    collect_aggregated_axioms,
    collect_private_traces,
)


def _load_log(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _get_session_info(data: Dict[str, Any]) -> Dict[str, Any]:
    return data.get("session_info") or data.get("session_metadata") or {}


def _extract_turn_logs(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    return data.get("turn_logs") or data.get("turns") or []


def _build_message_name(turn: Dict[str, Any]) -> str:
    module_id = turn.get("module_id")
    module_label = turn.get("module_label")
    if module_id and module_label:
        return f"{module_id}:{module_label}"

    team_id = turn.get("team_id")
    agent_name = turn.get("agent_name")
    if team_id and agent_name:
        return f"{team_id}:{agent_name}"

    agent_id = turn.get("agent_id") or turn.get("agent")
    if team_id and agent_id:
        return f"{team_id}:{agent_id}"

    return turn.get("speaker") or turn.get("name") or "unknown"


def _build_chat_history(turn_logs: List[Dict[str, Any]]) -> List[Any]:
    history: List[Any] = []
    for turn in turn_logs:
        content = turn.get("message_content") or turn.get("message") or ""
        if not content:
            continue
        name = _build_message_name(turn)
        history.append(SimpleNamespace(name=name, content=content))
    return history


def _extract_cfr_mode(session_info: Dict[str, Any]) -> str:
    metadata = session_info.get("metadata") if isinstance(session_info, dict) else None
    if isinstance(metadata, dict) and metadata.get("cfr_mode"):
        return str(metadata.get("cfr_mode"))
    if isinstance(session_info, dict) and session_info.get("cfr_mode"):
        return str(session_info.get("cfr_mode"))
    return "unknown"


def _extract_crisis_context(session_info: Dict[str, Any]) -> str:
    if not isinstance(session_info, dict):
        return ""
    return str(
        session_info.get("crisis_context")
        or session_info.get("topic")
        or session_info.get("prompt")
        or ""
    )


def _normalize_condition(value: Optional[str]) -> str:
    if not value:
        return "unknown"
    lowered = str(value).strip().lower()
    legacy_map = {
        "cfrnokg": "cfr_no_kg",
        "nocfr": "no_cfr_baseline",
        "kgcfrfull": "kg_cfr_full",
    }
    return legacy_map.get(lowered, lowered)


def _extract_condition(session_info: Dict[str, Any], path: Path) -> str:
    metadata = session_info.get("metadata") if isinstance(session_info, dict) else None
    if isinstance(metadata, dict):
        for key in ("condition_label", "cfr_mode", "condition_label_legacy"):
            value = metadata.get(key)
            if value:
                return _normalize_condition(value)

    if isinstance(session_info, dict):
        value = session_info.get("cfr_mode")
        if value:
            return _normalize_condition(value)

    stem = path.stem.lower()
    for token in (
        "kg_cfr_full",
        "cfr_no_kg",
        "no_cfr_baseline",
        "prompt_only",
        "prompt_only_idrag_on",
        "prompt_only_idrag_off",
        "knowledge_grounded",
    ):
        if token in stem:
            return token
    return "unknown"


def _is_excluded_name(path: Path) -> bool:
    stem = path.stem.lower()
    if stem.startswith("judge_") or stem.startswith("dis_"):
        return True
    if "judge_" in stem or "dis_" in stem:
        return True
    return False


def _has_full_debate(turn_logs: List[Dict[str, Any]]) -> bool:
    if not turn_logs:
        return False
    for turn in turn_logs:
        if turn.get("phase") in {"standoff", "debate"} and turn.get("message_content"):
            return True
    return False


def _collect_candidates(root: Path, prefixes: Optional[List[str]]) -> List[Tuple[str, Path]]:
    candidates: List[Tuple[str, Path]] = []
    prefix_set = [p.lower() for p in prefixes or []]

    for path in sorted(root.rglob("*.json")):
        if _is_excluded_name(path):
            continue
        stem = path.stem.lower()
        if prefix_set and not any(stem.startswith(p) for p in prefix_set):
            continue
        try:
            data = _load_log(path)
        except Exception:
            continue
        turn_logs = _extract_turn_logs(data)
        if not _has_full_debate(turn_logs):
            continue
        session_info = _get_session_info(data)
        condition = _extract_condition(session_info, path)
        candidates.append((condition, path))

    return candidates


def _pick_random_logs_by_condition(
    root: Path,
    per_condition: int,
    conditions: List[str],
    prefixes: Optional[List[str]],
    seed: Optional[int],
) -> List[Tuple[str, Path]]:
    candidates = _collect_candidates(root, prefixes)
    buckets: Dict[str, List[Path]] = {c: [] for c in conditions}
    for condition, path in candidates:
        if condition in buckets:
            buckets[condition].append(path)

    selections: List[Tuple[str, Path]] = []
    for condition in conditions:
        pool = sorted(buckets[condition])
        if not pool:
            continue
        if seed is None:
            rng = random.Random()
        else:
            condition_seed = seed + sum(ord(ch) for ch in condition)
            rng = random.Random(condition_seed)

        if per_condition >= len(pool):
            chosen = pool
        else:
            chosen = rng.sample(pool, per_condition)
        selections.extend((condition, path) for path in chosen)

    return selections


def _run_phase4_for_log(
    log_path: Path,
    synthesis_graph: Any,
) -> Dict[str, Any]:
    data = _load_log(log_path)
    session_info = _get_session_info(data)
    turn_logs = _extract_turn_logs(data)
    chat_history = _build_chat_history(turn_logs)

    aggregated_axioms = collect_aggregated_axioms({}, turn_logs=turn_logs)
    private_traces = collect_private_traces({}, turn_logs=turn_logs)

    synthesis_input: SynthesisState = {
        "crisis_context": _extract_crisis_context(session_info),
        "chat_history": chat_history,
        "private_thought_buffer": {},
        "cfr_mode": _extract_cfr_mode(session_info),
        "synthesis_iteration": 0,
        "synthesis_proposal": "",
        "synthesis_approved": False,
        "synthesis_final": "",
        "provenance_log": [],
        "aggregated_axioms": aggregated_axioms,
        "private_trace_snippets": private_traces,
        "eval_cf_payload": {},
        "self_correction_count": 0,
        "total_token_overhead": 0,
        "provenance_fidelity_scores": [],
        "phase4_execution_latency_ms": 0,
    }

    start_time = time.perf_counter()
    result = synthesis_graph.invoke(
        synthesis_input,
        config={"recursion_limit": max(25, MAX_SYNTHESIS_ITER * 6)},
    )
    latency_ms = int((time.perf_counter() - start_time) * 1000)
    if result:
        result["phase4_execution_latency_ms"] = latency_ms

    return {
        "session_id": session_info.get("session_id") or log_path.stem,
        "result": result,
    }


def _extract_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    result = row.get("result") or {}
    pf_scores = result.get("provenance_fidelity_scores") or []
    initial_pf = pf_scores[0] if pf_scores else None
    final_pf = pf_scores[-1] if pf_scores else None
    iterations = result.get("synthesis_iteration", 0)
    if isinstance(iterations, int):
        iterations = iterations + 1
    latency = result.get("phase4_execution_latency_ms")

    synthesis_final = result.get("synthesis_final")
    was_divergence = False
    if isinstance(synthesis_final, dict):
        was_divergence = synthesis_final.get("status") == "DIVERGENCE_REPORT"
    elif isinstance(synthesis_final, str) and "DIVERGENCE_REPORT" in synthesis_final:
        was_divergence = True

    pf_progression = "|".join(str(pf) for pf in pf_scores) if pf_scores else ""

    return {
        "condition": row.get("condition"),
        "session_id": row.get("session_id"),
        "initial_pf": initial_pf,
        "final_pf": final_pf,
        "iterations": iterations,
        "latency": latency,
        "was_divergence_report": was_divergence,
        "pf_progression": pf_progression,
    }


def run_ablation_experiment(
    archive_dir: Path,
    per_condition: int,
    conditions: List[str],
    prefixes: Optional[List[str]],
    seed: Optional[int],
    output_dir: Path,
    proposer_model: str,
    validator_model_a: str,
    validator_model_b: str,
) -> Dict[str, Any]:
    """Run ablation experiment with two different validator models."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get API and load proposer LLM
    api = get_philosopher_api()
    proposer_llm = api.get_llm(model_override=proposer_model)
    validator_llm_a = api.get_llm(model_override=validator_model_a)
    validator_llm_b = api.get_llm(model_override=validator_model_b)

    print(f"\n{'='*100}")
    print(f"ABLATION EXPERIMENT: Validator IQ")
    print(f"{'='*100}")
    print(f"Proposer Model: {proposer_model}")
    print(f"Validator A (Baseline): {validator_model_a}")
    print(f"Validator B (Expert): {validator_model_b}")
    print(f"Archive Dir: {archive_dir}")
    print(f"Per Condition: {per_condition}")
    print(f"{'='*100}\n")

    # Get sample logs (same for both runs)
    log_paths = _pick_random_logs_by_condition(
        archive_dir, per_condition, conditions, prefixes, seed
    )
    if not log_paths:
        print(f"[ERROR] No JSON logs found under: {archive_dir}")
        return {}

    print(f"[INFO] Loaded {len(log_paths)} log files for evaluation\n")

    # Run A: Flash Validator
    print(f"[LOG] Starting Run A (Validator={validator_model_a})...")
    synthesis_graph_a = build_synthesis_subgraph(
        llm=proposer_llm, validator_llm=validator_llm_a
    )
    rows_a: List[Dict[str, Any]] = []

    for idx, item in enumerate(log_paths, start=1):
        condition, path = item
        try:
            result = _run_phase4_for_log(path, synthesis_graph_a)
            result["condition"] = condition
            rows_a.append(_extract_metrics(result))
        except Exception as exc:
            print(f"[WARN] Failed on {path}: {exc}")

    # Run B: Pro Validator
    print(f"\n[LOG] Starting Run B (Validator={validator_model_b})...")
    synthesis_graph_b = build_synthesis_subgraph(
        llm=proposer_llm, validator_llm=validator_llm_b
    )
    rows_b: List[Dict[str, Any]] = []

    for idx, item in enumerate(log_paths, start=1):
        condition, path = item
        try:
            result = _run_phase4_for_log(path, synthesis_graph_b)
            result["condition"] = condition
            rows_b.append(_extract_metrics(result))
        except Exception as exc:
            print(f"[WARN] Failed on {path}: {exc}")

    # Write CSVs
    fieldnames = [
        "condition",
        "session_id",
        "initial_pf",
        "final_pf",
        "iterations",
        "latency",
        "was_divergence_report",
        "pf_progression",
    ]

    csv_a_path = output_dir / f"ablation_validator_iq_run_a_{validator_model_a.replace('-', '_')}.csv"
    csv_b_path = output_dir / f"ablation_validator_iq_run_b_{validator_model_b.replace('-', '_')}.csv"

    with csv_a_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_a:
            writer.writerow(row)

    with csv_b_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_b:
            writer.writerow(row)

    print(f"[SAVE] Run A results: {csv_a_path}")
    print(f"[SAVE] Run B results: {csv_b_path}")

    # Generate comparative summary
    summary = _generate_ablation_summary(rows_a, rows_b, validator_model_a, validator_model_b)
    summary_path = output_dir / f"ablation_validator_iq_summary.txt"

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(summary)

    print(f"[SAVE] Comparative summary: {summary_path}\n")

    return {
        "run_a": rows_a,
        "run_b": rows_b,
        "csv_a": csv_a_path,
        "csv_b": csv_b_path,
        "summary": summary_path,
    }


def _generate_ablation_summary(
    rows_a: List[Dict[str, Any]],
    rows_b: List[Dict[str, Any]],
    model_a: str,
    model_b: str,
) -> str:
    """Generate comparative summary of ablation experiment."""

    if not rows_a or not rows_b:
        return "ERROR: One or both runs produced no results.\n"

    def calc_stats(rows):
        n = len(rows)
        avg_initial_pf = sum(r.get("initial_pf", 0) or 0 for r in rows) / n
        avg_final_pf = sum(r.get("final_pf", 0) or 0 for r in rows) / n
        avg_iterations = sum(r.get("iterations", 1) or 1 for r in rows) / n
        divergence_count = sum(1 for r in rows if r.get("was_divergence_report"))
        approved_count = n - divergence_count

        return {
            "n": n,
            "avg_initial_pf": avg_initial_pf,
            "avg_final_pf": avg_final_pf,
            "avg_iterations": avg_iterations,
            "divergence_count": divergence_count,
            "approved_count": approved_count,
        }

    stats_a = calc_stats(rows_a)
    stats_b = calc_stats(rows_b)

    lines = []
    lines.append("=" * 120)
    lines.append("ABLATION STUDY: VALIDATOR IQ")
    lines.append("=" * 120)
    lines.append("")
    lines.append("HYPOTHESIS: A better (more capable) Validator LLM should approve higher-quality synthesis,")
    lines.append("resulting in fewer DIVERGENCE_REPORT cases and higher average provenance fidelity.")
    lines.append("")
    lines.append("=" * 120)
    lines.append("RESULTS COMPARISON")
    lines.append("=" * 120)
    lines.append("")

    lines.append(f"{'Metric':<30} | {'Run A (':<40} | {'Run B (':<40}")
    lines.append(f"{'':30} | {model_a:<38} | {model_b:<38}")
    lines.append("-" * 120)

    lines.append(f"{'Sample Size':<30} | {stats_a['n']:<38} | {stats_b['n']:<38}")
    lines.append(f"{'Avg Initial PF':<30} | {stats_a['avg_initial_pf']:<38.3f} | {stats_b['avg_initial_pf']:<38.3f}")
    lines.append(f"{'Avg Final PF':<30} | {stats_a['avg_final_pf']:<38.3f} | {stats_b['avg_final_pf']:<38.3f}")
    lines.append(f"{'Avg Iterations':<30} | {stats_a['avg_iterations']:<38.2f} | {stats_b['avg_iterations']:<38.2f}")
    lines.append(f"{'Approved':<30} | {stats_a['approved_count']}/{stats_a['n']:<35} ({100*stats_a['approved_count']/stats_a['n']:.1f}%) | {stats_b['approved_count']}/{stats_b['n']:<35} ({100*stats_b['approved_count']/stats_b['n']:.1f}%)")
    lines.append(f"{'DIVERGENCE_REPORT':<30} | {stats_a['divergence_count']}/{stats_a['n']:<35} ({100*stats_a['divergence_count']/stats_a['n']:.1f}%) | {stats_b['divergence_count']}/{stats_b['n']:<35} ({100*stats_b['divergence_count']/stats_b['n']:.1f}%)")

    lines.append("")
    lines.append("=" * 120)
    lines.append("INTERPRETATION")
    lines.append("=" * 120)
    lines.append("")

    if stats_b["approved_count"] > stats_a["approved_count"]:
        lines.append(f"✓ Run B (Expert Validator) approved {stats_b['approved_count'] - stats_a['approved_count']} MORE cases.")
        lines.append(f"  This suggests the better validator is less strict (or more lenient).")
    elif stats_b["approved_count"] < stats_a["approved_count"]:
        lines.append(f"✗ Run B (Expert Validator) approved {stats_a['approved_count'] - stats_b['approved_count']} FEWER cases.")
        lines.append(f"  This suggests the better validator is stricter.")
    else:
        lines.append(f"≈ Both validators approved the same number of cases.")

    if stats_b["avg_final_pf"] > stats_a["avg_final_pf"]:
        lines.append(f"✓ Run B showed higher avg final PF ({stats_b['avg_final_pf']:.3f} vs {stats_a['avg_final_pf']:.3f}).")
    elif stats_b["avg_final_pf"] < stats_a["avg_final_pf"]:
        lines.append(f"✗ Run B showed lower avg final PF ({stats_b['avg_final_pf']:.3f} vs {stats_a['avg_final_pf']:.3f}).")
    else:
        lines.append(f"≈ Both validators resulted in same avg final PF.")

    lines.append("")
    lines.append("=" * 120)

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ablation Study: Validator IQ Experiment")
    parser.add_argument(
        "--archive-dir",
        default="logs/archive/article2",
        help="Root directory containing archived JSON logs",
    )
    parser.add_argument(
        "--per-condition",
        type=int,
        default=30,
        help="Sample size per condition",
    )
    parser.add_argument(
        "--conditions",
        default="kg_cfr_full",
        help="Comma-separated list of conditions to include",
    )
    parser.add_argument(
        "--prefixes",
        default=None,
        help="Comma-separated list of session_id prefixes to include",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        default="analysis/outputs/ablation_validator_iq",
        help="Output directory for results",
    )
    parser.add_argument(
        "--proposer-model",
        default="gemini-2.5-flash-lite",
        help="Model for Proposer (both runs)",
    )
    parser.add_argument(
        "--validator-model-a",
        default="gemini-2.5-flash-lite",
        help="Model for Validator in Run A (Baseline)",
    )
    parser.add_argument(
        "--validator-model-b",
        default="gemini-2.5-pro",
        help="Model for Validator in Run B (Expert)",
    )

    args = parser.parse_args()

    archive_dir = Path(args.archive_dir)
    if not archive_dir.exists():
        print(f"[ERROR] Archive directory not found: {archive_dir}")
        return 1

    prefixes = None
    if args.prefixes:
        prefixes = [p.strip() for p in args.prefixes.split(",") if p.strip()]

    conditions = [
        _normalize_condition(c.strip())
        for c in args.conditions.split(",")
        if c.strip()
    ]

    if not conditions:
        print("[ERROR] No conditions specified.")
        return 1

    output_dir = Path(args.output_dir)

    run_ablation_experiment(
        archive_dir=archive_dir,
        per_condition=args.per_condition,
        conditions=conditions,
        prefixes=prefixes,
        seed=args.seed,
        output_dir=output_dir,
        proposer_model=args.proposer_model,
        validator_model_a=args.validator_model_a,
        validator_model_b=args.validator_model_b,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
