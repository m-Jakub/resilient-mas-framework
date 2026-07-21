"""
Compute deterministic v5 metrics (CC, DIS, SR) for AEGIS/DRAU logs.

- CC: cosine similarity between plan decision fields and public response.
- DIS: NLI-based entailment/contradiction coverage of opponent claims.
- SR: max cosine similarity to own previous three turns, normalized by log length.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from src.common.utils.claim_segmenter import extract_claims


_CC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"


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


def _get_cc_model() -> SentenceTransformer:
    if not hasattr(_get_cc_model, "_model"):
        _get_cc_model._model = SentenceTransformer(_CC_MODEL_NAME)
    return _get_cc_model._model


def _release_cc_model() -> None:
    if hasattr(_get_cc_model, "_model"):
        delattr(_get_cc_model, "_model")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _get_nli_model() -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    if not hasattr(_get_nli_model, "_tokenizer"):
        _get_nli_model._tokenizer = AutoTokenizer.from_pretrained(_NLI_MODEL_NAME)
        _get_nli_model._model = AutoModelForSequenceClassification.from_pretrained(_NLI_MODEL_NAME)
        _get_nli_model._model.eval()
    return _get_nli_model._tokenizer, _get_nli_model._model


def _release_nli_model() -> None:
    if hasattr(_get_nli_model, "_tokenizer"):
        delattr(_get_nli_model, "_tokenizer")
    if hasattr(_get_nli_model, "_model"):
        delattr(_get_nli_model, "_model")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if vec_a is None or vec_b is None:
        return 0.0
    if len(vec_a) == 0 or len(vec_b) == 0:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    denom = max(norm_a * norm_b, 1e-8)
    return max(0.0, min(1.0, dot / denom))


def _extract_turns(log: Dict[str, Any]) -> List[Dict[str, Any]]:
    turns = log.get("turn_logs", [])
    return [t for t in turns if t.get("phase") in {"debate", "standoff"}]


def _parse_private_strategy(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _plan_text_from_strategy(strategy: Dict[str, Any]) -> Optional[str]:
    if not strategy:
        return None
    target_id = strategy.get("target_opponent_claim_id")
    attack_surface = strategy.get("attack_surface")
    strategic_intent = strategy.get("strategic_intent")
    if not target_id or not attack_surface or not strategic_intent:
        return None
    return f"{target_id} | {attack_surface} | {strategic_intent}"


def _find_last_opponent_turns(turns: List[Dict[str, Any]], idx: int, module_id: str) -> List[Dict[str, Any]]:
    opponents: List[Dict[str, Any]] = []
    seen_ids = set()
    for j in range(idx - 1, -1, -1):
        prev_id = turns[j].get("module_id") or turns[j].get("team_id")
        if prev_id and prev_id != module_id and prev_id not in seen_ids:
            opponents.append(turns[j])
            seen_ids.add(prev_id)
        if len(opponents) >= 2:
            break
    return opponents


def _collect_opponent_claims(turn: Dict[str, Any], turns: List[Dict[str, Any]], idx: int) -> List[str]:
    metadata = turn.get("metadata") or {}
    payload = metadata.get("opponent_claims") or []
    claims: List[str] = []
    if isinstance(payload, list) and payload:
        for entry in payload:
            for claim in entry.get("claims", []) or []:
                text = claim.get("claim_text") if isinstance(claim, dict) else None
                if text:
                    claims.append(text)
# Filter placeholder texts that appear when no opponent has spoken yet.
        claims = [t for t in claims if t.lower().strip() not in _DIS_PLACEHOLDER_CLAIMS]

        if claims:
            return claims

        module_id = turn.get("module_id") or turn.get("team_id") or ""
        for opp in _find_last_opponent_turns(turns, idx, module_id):
            msg = opp.get("message_content") or ""
            claims.extend([c["claim_text"] for c in extract_claims(msg)])

        # Final filter: remove placeholders from fallback path too.
        return [t for t in claims if t.lower().strip() not in _DIS_PLACEHOLDER_CLAIMS]


def _compute_cc(turns: List[Dict[str, Any]]) -> Dict[int, float]:
    plan_texts: List[str] = []
    response_texts: List[str] = []
    index_map: List[Tuple[int, int]] = []

    for idx, turn in enumerate(turns):
        metadata = turn.get("metadata") or {}
        # Field renamed: orchestrator logs JSON as 'private_strategy_json'.
        # Fall back to legacy 'private_strategy' for logs produced before the rename.
        strategy = _parse_private_strategy(
            metadata.get("private_strategy_json")
            or metadata.get("private_strategy")
            or ""
        )
        plan_text = _plan_text_from_strategy(strategy)
        response = turn.get("message_content") or ""
        if not plan_text or not response:
            continue
        pos = len(plan_texts)
        plan_texts.append(plan_text)
        response_texts.append(response)
        index_map.append((idx, pos))

    if not plan_texts:
        return {}

    model = _get_cc_model()
    plan_vecs = model.encode(plan_texts, normalize_embeddings=True)
    resp_vecs = model.encode(response_texts, normalize_embeddings=True)

    cc_scores: Dict[int, float] = {}
    for idx, pos in index_map:
        cc_scores[idx] = _cosine_similarity(plan_vecs[pos], resp_vecs[pos])

    return cc_scores


def _compute_sr(turns: List[Dict[str, Any]]) -> Dict[int, float]:
    responses: List[str] = [t.get("message_content") or "" for t in turns]
    model = _get_cc_model()
    resp_vecs = model.encode(responses, normalize_embeddings=True)

    sr_scores: Dict[int, float] = {}
    history_by_module: Dict[str, List[int]] = {}

    for idx, turn in enumerate(turns):
        response = responses[idx]
        module_id = turn.get("module_id") or turn.get("team_id") or ""
        history = history_by_module.get(module_id, [])
        if not response:
            continue

        if not history:
            history_by_module.setdefault(module_id, []).append(idx)
            continue

        prev_indices = history[-3:]
        sims = [
            _cosine_similarity(resp_vecs[idx], resp_vecs[p])
            for p in prev_indices
        ]
        base = max(sims) if sims else 0.0
        token_count = max(1, len(response.split()))
        denom = math.log(token_count + 2)
        sr_scores[idx] = base / denom if denom > 0 else base
        history_by_module.setdefault(module_id, []).append(idx)

    return sr_scores


_DIS_PLACEHOLDER_CLAIMS = {
    "(no recent message available)",
    "no recent message available",
    "n/a",
    "",
}
_NLI_CONFIDENCE_THRESHOLD = 0.75


def _predict_nli_label(premise: str, hypothesis: str) -> tuple[str, float]:
    """Returns (label, confidence). Only ENTAILMENT above threshold counts as a DIS hit."""
    tokenizer, model = _get_nli_model()
    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    pred_id = int(torch.argmax(logits, dim=-1).item())
    confidence = float(probs[0, pred_id].item())
    id2label = model.config.id2label or {}
    return str(id2label.get(pred_id, "neutral")).lower(), confidence


def _compute_dis(turns: List[Dict[str, Any]]) -> Dict[int, float]:
    dis_scores: Dict[int, float] = {}
    for idx, turn in enumerate(turns):
        response = turn.get("message_content") or ""
        if not response:
            continue
        claims = _collect_opponent_claims(turn, turns, idx)
        if not claims:
            continue

        hit = 0
        for claim in claims:
            label, confidence = _predict_nli_label(claim, response)
            # Count only high-confidence ENTAILMENT as a DIS hit.
            # Contradiction means the response disputes the claim (valid engagement)
            # but using both entail+contradict trivially inflates the score to ~1.0.
            # Entailment above threshold means the response genuinely addresses the claim.
            if "entail" in label and confidence >= _NLI_CONFIDENCE_THRESHOLD:
                hit += 1
        # claims is guaranteed non-empty here (we `continue`d above if empty)
        dis_scores[idx] = hit / len(claims)

    return dis_scores


def compute_v5_metrics(log_path: Path) -> Dict[str, Any]:
    with log_path.open("r", encoding="utf-8") as f:
        log = json.load(f)

    turns = _extract_turns(log)
    cc_scores = _compute_cc(turns)
    sr_scores = _compute_sr(turns)
    _release_cc_model()
    dis_scores = _compute_dis(turns)
    _release_nli_model()

    per_turn_rows: List[Dict[str, Any]] = []
    cc_values: List[float] = []
    dis_values: List[float] = []
    sr_values: List[float] = []

    for idx, turn in enumerate(turns):
        turn_number = turn.get("turn_number")
        module_id = turn.get("module_id")
        module_label = turn.get("module_label")

        cc = cc_scores.get(idx)
        dis = dis_scores.get(idx)
        sr = sr_scores.get(idx)

        if cc is not None:
            cc_values.append(cc)
        if dis is not None:
            dis_values.append(dis)
        if sr is not None:
            sr_values.append(sr)

        per_turn_rows.append({
            "turn_number": turn_number,
            "module_id": module_id,
            "module_label": module_label,
            "cc": cc,
            "dis": dis,
            "sr": sr,
        })

    summary = {
        "turns_scored": len(per_turn_rows),
        "avg_cc": sum(cc_values) / len(cc_values) if cc_values else None,
        "avg_dis": sum(dis_values) / len(dis_values) if dis_values else None,
        "avg_sr": sum(sr_values) / len(sr_values) if sr_values else None,
    }

    return {
        "log_path": str(log_path),
        "session_id": log.get("session_info", {}).get("session_id"),
        "summary": summary,
        "per_turn": per_turn_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute v5 metrics over AEGIS/DRAU logs")
    parser.add_argument("--logs", required=True, help="Path to a log JSON file or a directory")
    parser.add_argument(
        "--output-dir",
        default="analysis/outputs/e2/metrics",
        help="Output directory",
    )
    args = parser.parse_args()

    logs_path = Path(args.logs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for log_path in _load_log_paths(logs_path):
        result = compute_v5_metrics(log_path)
        session_id = result.get("session_id") or log_path.stem

        summary = result.get("summary", {})
        print(
            f"[v5] {session_id} | CC={summary.get('avg_cc')} "
            f"DIS={summary.get('avg_dis')} SR={summary.get('avg_sr')}"
        )

        for row in result["per_turn"]:
            row = dict(row)
            row["session_id"] = session_id
            all_rows.append(row)

        summary = dict(result["summary"])
        summary["session_id"] = session_id
        summaries.append(summary)

    _write_csv(output_dir / "v5_metrics_turns.csv", all_rows)
    _write_csv(output_dir / "v5_metrics_summary.csv", summaries)

    print(f"Saved v5 metrics outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
