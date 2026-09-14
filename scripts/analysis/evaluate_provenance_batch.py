"""
Batch evaluator for Phase 4 provenance validation on archived logs.

Samples N random JSON logs, runs Phase 4 synthesis validation, and writes
summary metrics to a CSV file.
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


def _collect_candidates(
    root: Path,
    prefixes: Optional[List[str]],
) -> List[Tuple[str, Path]]:
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


def _pick_random_logs(
    root: Path,
    sample_size: int,
    seed: Optional[int],
    conditions: Optional[List[str]],
    prefixes: Optional[List[str]],
) -> List[Tuple[str, Path]]:
    candidates = _collect_candidates(root, prefixes)
    if conditions:
        allowed = {c.lower() for c in conditions}
        candidates = [item for item in candidates if item[0] in allowed]

    if not candidates:
        return []

    rng = random.Random(seed)
    if sample_size >= len(candidates):
        return candidates
    return rng.sample(candidates, sample_size)


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


def _extract_critique_from_eval_cf(eval_cf_result: Dict[str, Any]) -> str:
    """Extract critique/reasoning text from eval_cf_result."""
    if not eval_cf_result:
        return ""

    critique_parts = []

    # Try common keys for critique
    for key in ("critique", "feedback", "reasoning", "explanation", "corrected_synthesis"):
        if key in eval_cf_result:
            val = eval_cf_result[key]
            if val and isinstance(val, str):
                critique_parts.append(val)

    # Also include unsupported sentences if present
    if "unsupported_sentences" in eval_cf_result:
        unsupported = eval_cf_result["unsupported_sentences"]
        if unsupported:
            if isinstance(unsupported, list):
                critique_parts.append("Unsupported: " + "; ".join(unsupported))
            elif isinstance(unsupported, str):
                critique_parts.append(f"Unsupported: {unsupported}")

    return " | ".join(critique_parts)


def _collect_audit_trail_data(row: Dict[str, Any]) -> Dict[str, Any]:
    """Extract audit trail information from Phase 4 result."""
    result = row.get("result") or {}
    provenance_log = result.get("provenance_log") or []

    initial_synthesis = ""
    final_synthesis = result.get("synthesis_final", "")
    critiques_list = []

    # Extract initial synthesis and all critiques from provenance log
    for i, record in enumerate(provenance_log):
        if i == 0:
            initial_synthesis = record.get("proposed_text", "")

        eval_cf_result = record.get("eval_cf_result", {})
        auditor_verdict = record.get("auditor_verdict", "")

        # Only collect critiques for iterations that were rejected
        if auditor_verdict == "rejected":
            critique_text = _extract_critique_from_eval_cf(eval_cf_result)
            if critique_text:
                critiques_list.append({
                    "iteration": i,
                    "critique": critique_text,
                    "proposed_before_critique": record.get("proposed_text", "")
                })

    return {
        "initial_synthesis": initial_synthesis,
        "final_synthesis": final_synthesis,
        "critiques": critiques_list,
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

    # Calculate pf_progression as pipe-separated list
    pf_progression = "|".join(str(pf) for pf in pf_scores) if pf_scores else ""

    # Calculate total critiques character length
    provenance_log = result.get("provenance_log") or []
    total_critiques_length = 0
    for record in provenance_log:
        eval_cf_result = record.get("eval_cf_result", {})
        critique = _extract_critique_from_eval_cf(eval_cf_result)
        total_critiques_length += len(critique)

    return {
        "condition": row.get("condition"),
        "prefix": row.get("prefix"),
        "session_id": row.get("session_id"),
        "initial_pf": initial_pf,
        "final_pf": final_pf,
        "iterations": iterations,
        "latency": latency,
        "was_divergence_report": was_divergence,
        "pf_progression": pf_progression,
        "total_critiques_length": total_critiques_length,
    }


def _extract_prefix(path: Path, prefixes: Optional[List[str]]) -> Optional[str]:
    if not prefixes:
        return None
    stem = path.stem.lower()
    for prefix in prefixes:
        if stem.startswith(prefix.lower()):
            return prefix
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch Phase 4 provenance evaluation")
    parser.add_argument(
        "--archive-dir",
        default="logs/archive/article2",
        help="Root directory containing archived JSON logs",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Total number of random JSON logs to evaluate",
    )
    parser.add_argument(
        "--per-condition",
        type=int,
        default=None,
        help="Sample size per condition (overrides --sample-size)",
    )
    parser.add_argument(
        "--conditions",
        default="no_cfr_baseline,cfr_no_kg,kg_cfr_full",
        help="Comma-separated list of conditions to include",
    )
    parser.add_argument(
        "--prefixes",
        default=None,
        help="Comma-separated list of session_id prefixes to include",
    )
    parser.add_argument(
        "--proposer-model",
        default=None,
        help="Override proposer model name (e.g., gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--validator-model",
        default=None,
        help="Override validator model name (e.g., gemini-3.1-pro-preview)",
    )
    parser.add_argument(
        "--validator-threshold",
        type=float,
        default=None,
        help="Override provenance fidelity threshold (e.g., 0.95)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: analysis/outputs/provenance_batch_YYYYMMDD_HHMMSS.csv)",
    )
    parser.add_argument(
        "--provenance-log-output",
        default=None,
        help="Output JSONL path for per-iteration provenance logs",
    )
    args = parser.parse_args()

    archive_dir = Path(args.archive_dir)
    if not archive_dir.exists():
        print(f"[ERROR] Archive directory not found: {archive_dir}")
        return 1

    prefixes = None
    if args.prefixes:
        prefixes = [p.strip() for p in args.prefixes.split(",") if p.strip()]

    conditions = []
    if args.conditions:
        conditions = [_normalize_condition(c.strip()) for c in args.conditions.split(",") if c.strip()]

    if args.per_condition is not None:
        if not conditions:
            print("[ERROR] --per-condition requires at least one condition.")
            return 1
        log_paths = _pick_random_logs_by_condition(
            archive_dir,
            args.per_condition,
            conditions,
            prefixes,
            args.seed,
        )
        if not log_paths:
            print(f"[ERROR] No JSON logs found under: {archive_dir}")
            return 1

        counts = {c: 0 for c in conditions}
        for condition, _ in log_paths:
            counts[condition] += 1
        for condition in conditions:
            if counts[condition] < args.per_condition:
                print(
                    f"[WARN] {condition}: requested {args.per_condition}, "
                    f"found {counts[condition]}"
                )
    else:
        if args.sample_size is None:
            print("[ERROR] Provide --sample-size or --per-condition.")
            return 1
        log_paths = _pick_random_logs(
            archive_dir,
            args.sample_size,
            args.seed,
            conditions,
            prefixes,
        )
        if not log_paths:
            print(f"[ERROR] No JSON logs found under: {archive_dir}")
            return 1

    api = get_philosopher_api()

    # Allow overriding proposer and validator models from CLI
    proposer_model = args.proposer_model
    validator_model = args.validator_model

    if proposer_model:
        proposer_llm = api.get_llm(model_override=proposer_model)
    else:
        proposer_llm = api.get_llm()

    if validator_model:
        validator_llm = api.get_llm(model_override=validator_model)
    else:
        validator_llm = None

    synthesis_graph = build_synthesis_subgraph(
        llm=proposer_llm,
        validator_llm=validator_llm,
        validator_threshold=args.validator_threshold,
    )

    rows: List[Dict[str, Any]] = []
    audit_trails: Dict[str, Any] = {}
    provenance_records: List[Dict[str, Any]] = []

    for idx, item in enumerate(log_paths, start=1):
        condition, path = item
        print(f"[{idx}/{len(log_paths)}] Evaluating: {path}")
        try:
            result = _run_phase4_for_log(path, synthesis_graph)
            result["condition"] = condition
            result["prefix"] = _extract_prefix(path, prefixes)

            # Extract metrics for CSV
            metrics_row = _extract_metrics(result)
            rows.append(metrics_row)

            # Extract audit trail data
            session_id = result.get("session_id")
            audit_trail = _collect_audit_trail_data(result)
            audit_trails[session_id] = {
                "condition": condition,
                "prefix": _extract_prefix(path, prefixes),
                "initial_synthesis": audit_trail["initial_synthesis"],
                "final_synthesis": audit_trail["final_synthesis"],
                "critiques": audit_trail["critiques"],
            }

            # Store per-iteration provenance log for auto-labeling
            prov_log = (result.get("result") or {}).get("provenance_log") or []
            for record in prov_log:
                eval_cf_result = record.get("eval_cf_result") or {}
                unsupported = eval_cf_result.get("unsupported_sentences") or []
                if isinstance(unsupported, str):
                    unsupported = [unsupported]
                provenance_records.append(
                    {
                        "session_id": session_id,
                        "condition": condition,
                        "prefix": _extract_prefix(path, prefixes),
                        "iteration": record.get("iteration"),
                        "auditor_verdict": record.get("auditor_verdict"),
                        "provenance_fidelity": record.get("provenance_fidelity"),
                        "proposed_text": record.get("proposed_text"),
                        "unsupported_sentences": unsupported,
                    }
                )
        except Exception as exc:
            print(f"[WARN] Failed on {path}: {exc}")

    if not rows:
        print("[ERROR] No results collected.")
        return 1

    if args.output:
        output_path = Path(args.output)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("analysis/outputs") / f"provenance_batch_{stamp}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "condition",
        "prefix",
        "session_id",
        "initial_pf",
        "final_pf",
        "iterations",
        "latency",
        "was_divergence_report",
        "pf_progression",
        "total_critiques_length",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Saved results to: {output_path}")

    # Write per-iteration provenance log JSONL for auto-labeling
    if args.provenance_log_output:
        provenance_path = Path(args.provenance_log_output)
    else:
        provenance_path = output_path.with_stem(
            output_path.stem.replace("provenance_batch", "provenance_log")
        ).with_suffix(".jsonl")

    if provenance_records:
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        with provenance_path.open("w", encoding="utf-8") as handle:
            for record in provenance_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved provenance log to: {provenance_path}")

    # Write audit trails to text file
    audit_path = output_path.with_stem(output_path.stem.replace("provenance_batch", "audit_trails")).with_suffix(".txt")
    with audit_path.open("w", encoding="utf-8") as f:
        f.write("=" * 120 + "\n")
        f.write("PHASE 4 (APVL) - DETAILED AUDIT TRAILS\n")
        f.write("=" * 120 + "\n\n")

        for session_id, trail in audit_trails.items():
            f.write(f"SESSION ID: {session_id}\n")
            f.write(f"Condition: {trail['condition']}\n")
            if trail['prefix']:
                f.write(f"Prefix: {trail['prefix']}\n")
            f.write("-" * 120 + "\n\n")

            # Initial synthesis
            f.write("INITIAL SYNTHESIS (Baseline):\n")
            f.write("-" * 40 + "\n")
            f.write(trail["initial_synthesis"] + "\n\n")

            # Critiques from validator (if any)
            if trail["critiques"]:
                f.write("VALIDATOR CRITIQUES (Rejected Iterations):\n")
                f.write("-" * 40 + "\n")
                for critique_item in trail["critiques"]:
                    f.write(f"\n[Iteration {critique_item['iteration']}]\n")
                    f.write("Proposed Before Critique:\n")
                    f.write(critique_item["proposed_before_critique"] + "\n")
                    f.write(f"\nValidator Critique:\n")
                    f.write(critique_item["critique"] + "\n")
                f.write("\n")
            else:
                f.write("No validator critiques (approved on first iteration)\n\n")

            # Final synthesis
            f.write("FINAL SYNTHESIS (Approved):\n")
            f.write("-" * 40 + "\n")
            final = trail["final_synthesis"]
            if isinstance(final, dict):
                f.write(json.dumps(final, indent=2, ensure_ascii=False) + "\n")
            else:
                f.write(str(final) + "\n")

            f.write("\n" + "=" * 120 + "\n\n")

    print(f"Saved audit trails to: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
