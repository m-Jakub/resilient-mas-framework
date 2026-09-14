"""Build a ready-to-annotate CSV from provenance batch + audit trails."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def _parse_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    text = _collapse_whitespace(text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _normalize_sentence(text: str) -> str:
    return _collapse_whitespace(text).strip()


def _classify_claim_scope(sentence: str) -> str:
    lowered = sentence.lower()
    separators = [
        ";",
        " and ",
        " oraz ",
        " but ",
        " ale ",
        " jednak ",
        " natomiast ",
        " podczas gdy ",
        " while ",
    ]
    if any(token in lowered for token in separators):
        return "multi"
    return "single"


def _load_provenance_jsonl(path: Path) -> Dict[str, List[Dict[str, object]]]:
    sessions: Dict[str, List[Dict[str, object]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        session_id = record.get("session_id")
        if not session_id:
            continue
        sessions.setdefault(str(session_id), []).append(record)
    for session_id, records in sessions.items():
        records.sort(key=lambda r: int(r.get("iteration", 0)))
    return sessions


def _merge_provenance_logs(paths: List[Path]) -> Dict[str, List[Dict[str, object]]]:
    merged: Dict[str, List[Dict[str, object]]] = {}
    for path in paths:
        if not path.exists():
            continue
        sessions = _load_provenance_jsonl(path)
        for session_id, records in sessions.items():
            if session_id not in merged:
                merged[session_id] = records
    return merged


def _auto_label_for_sentence(sentence: str, record: Dict[str, object]) -> str:
    unsupported = record.get("unsupported_sentences") or []
    if isinstance(unsupported, str):
        unsupported = [unsupported]
    unsupported_set = {_normalize_sentence(str(item)) for item in unsupported}
    norm_sentence = _normalize_sentence(sentence)
    if norm_sentence in unsupported_set:
        return "unsupported"
    return "supported"


def _infer_provenance_log_path(provenance_csv: Path) -> Optional[Path]:
    if provenance_csv.stem.startswith("provenance_batch_"):
        suffix = provenance_csv.stem.replace("provenance_batch_", "")
        candidate = provenance_csv.with_name(f"provenance_log_{suffix}.jsonl")
        if candidate.exists():
            return candidate
    return None


def _parse_audit_trails(path: Path) -> Dict[str, Dict[str, str]]:
    sessions: Dict[str, Dict[str, str]] = {}
    current_id: Optional[str] = None
    current: Dict[str, str] = {}
    state: Optional[str] = None
    buffer: List[str] = []

    def _flush_buffer() -> None:
        nonlocal buffer, state, current
        if state and buffer:
            content = "\n".join(buffer).strip()
            current[state] = content
        buffer = []
        state = None

    def _finalize_session() -> None:
        nonlocal current_id, current
        if current_id:
            sessions[current_id] = dict(current)
        current_id = None
        current = {}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if line.startswith("SESSION ID:"):
            _flush_buffer()
            _finalize_session()
            current_id = line.split(":", 1)[1].strip()
            continue
        if line.startswith("Condition:"):
            current["condition"] = line.split(":", 1)[1].strip()
            continue
        if line.startswith("Prefix:"):
            current["prefix"] = line.split(":", 1)[1].strip()
            continue
        if line.startswith("INITIAL SYNTHESIS"):
            _flush_buffer()
            state = "initial"
            continue
        if line.startswith("FINAL SYNTHESIS"):
            _flush_buffer()
            state = "final"
            continue
        if line.startswith("VALIDATOR CRITIQUES"):
            _flush_buffer()
            continue
        if line.startswith("=") and len(line) >= 10:
            _flush_buffer()
            continue
        if line.startswith("-") and len(line) >= 10:
            continue
        if state:
            buffer.append(line)

    _flush_buffer()
    _finalize_session()
    return sessions


def _final_text_from_audit(entry: Dict[str, str]) -> str:
    final_text = entry.get("final", "").strip()
    if not final_text:
        return ""

    if final_text.startswith("{") and final_text.endswith("}"):
        try:
            payload = json.loads(final_text)
        except json.JSONDecodeError:
            return final_text
        if isinstance(payload, dict) and payload.get("status") == "DIVERGENCE_REPORT":
            return str(payload.get("last_attempted_synthesis", "")).strip()
    return final_text


def main() -> int:
    parser = argparse.ArgumentParser(description="Build annotation CSV from audit trails")
    parser.add_argument(
        "--audit-trails",
        default="analysis/outputs/audit_trails_20260502_154449.txt",
        help="Path to audit trails txt",
    )
    parser.add_argument(
        "--provenance-csv",
        default="analysis/outputs/provenance_batch_20260502_154449.csv",
        help="Path to provenance batch csv",
    )
    parser.add_argument(
        "--provenance-log-jsonl",
        default="analysis/outputs/provenance_log_20260502_154449.jsonl",
        help="Path to provenance log JSONL (per iteration)",
    )
    parser.add_argument(
        "--prefixes",
        default="exp_e2_20260314_052549",
        help="Comma-separated list of session_id prefixes",
    )
    parser.add_argument(
        "--condition",
        default="kg_cfr_full",
        help="Condition to include",
    )
    parser.add_argument("--sample-size", type=int, default=10, help="Sample size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        default="analysis/outputs/annotation_ready.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    audit_path = Path(args.audit_trails)
    provenance_path = Path(args.provenance_csv)
    provenance_log_path = Path(args.provenance_log_jsonl)
    if not audit_path.exists():
        raise FileNotFoundError(f"Missing audit trails file: {audit_path}")
    if not provenance_path.exists():
        raise FileNotFoundError(f"Missing provenance CSV: {provenance_path}")
    if not provenance_log_path.exists():
        inferred = _infer_provenance_log_path(provenance_path)
        if inferred:
            provenance_log_path = inferred
        else:
            raise FileNotFoundError(
                f"Missing provenance log JSONL: {provenance_log_path}\n"
                "Run evaluate_provenance_batch.py with --provenance-log-output to create it."
            )

    prefixes = set(_parse_list(args.prefixes))
    df = pd.read_csv(provenance_path)
    if args.condition:
        df = df[df["condition"] == args.condition]
    if prefixes:
        df = df[df["prefix"].isin(prefixes)]

    session_ids = df["session_id"].dropna().unique().tolist()
    if not session_ids:
        raise ValueError("No sessions found for the provided filters.")

    rng = random.Random(args.seed)
    if len(session_ids) > args.sample_size:
        session_ids = rng.sample(session_ids, args.sample_size)

    sessions = _parse_audit_trails(audit_path)
    provenance_logs = _load_provenance_jsonl(provenance_log_path)
    rows: List[Dict[str, str]] = []

    for session_id in session_ids:
        entry = sessions.get(session_id)
        if not entry:
            continue

        prov_records = provenance_logs.get(session_id)
        if not prov_records:
            outputs_dir = provenance_log_path.parent
            candidates = sorted(
                outputs_dir.glob("provenance_log_*.jsonl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            provenance_logs = _merge_provenance_logs(candidates)
            prov_records = provenance_logs.get(session_id)
        if not prov_records:
            raise ValueError(
                "Missing provenance records for session: "
                f"{session_id}.\n"
                "Use --provenance-log-jsonl pointing to the JSONL produced by the same batch run."
            )

        baseline_record = prov_records[0]
        final_record = prov_records[-1]

        baseline_text = entry.get("initial", "").strip()
        final_text = _final_text_from_audit(entry)
        condition = entry.get("condition", args.condition)
        prefix = entry.get("prefix", "")

        for synth_type, text in (("baseline", baseline_text), ("apg", final_text)):
            record = baseline_record if synth_type == "baseline" else final_record
            sentences = _split_sentences(text)
            for idx, sentence in enumerate(sentences, start=1):
                auto_label = _auto_label_for_sentence(sentence, record)
                auto_binary = "1" if auto_label == "supported" else "0"
                rows.append(
                    {
                        "session_id": session_id,
                        "condition": condition,
                        "prefix": prefix,
                        "synthesis_type": synth_type,
                        "sentence_id": f"{synth_type}-{idx:02d}",
                        "sentence_text": sentence,
                        "claim_scope": _classify_claim_scope(sentence),
                        "auto_label": auto_label,
                        "auto_binary": auto_binary,
                        "label_A": "",
                        "label_B": "",
                        "adjudicated_label": "",
                        "human_binary_label": "",
                        "evidence_type": "",
                        "evidence_pointer": "",
                        "notes": "",
                    }
                )

    if not rows:
        raise ValueError("No annotation rows produced. Check your filters.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote annotation sheet: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
