"""
Offline Webis-aligned judge (LLM-as-a-Judge).

Scores arguments with Webis-ArgQuality-style criteria:
- Clarity
- Cogency
- Relevance

This is an offline, reproducible evaluator intended for E1 calibration
and E2 scoring pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import argparse
import csv
import json
import re
import sys
import warnings

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from src.common.api_abstraction import get_philosopher_api
from src.common.llm_retry import invoke_with_retry

_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")
_SCORE_RE = re.compile(r"SCORE\s*:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

MIN_JSON_RATIO_WARN = 0.8
MIN_JSON_RATIO_FAIL = 0.5
MIN_SUBSCORE_RATIO_WARN = 0.8
MIN_SUBSCORE_RATIO_FAIL = 0.5


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass
class JudgeScore:
    clarity: Optional[float]
    cogency: Optional[float]
    relevance: Optional[float]
    overall: Optional[float]


class WebisAlignedOfflineJudge:
    """LLM-as-a-Judge aligned with ArgQuality-style criteria."""

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = weights or {
            "clarity": 0.3,
            "cogency": 0.3,
            "relevance": 0.4,
        }
        api = get_philosopher_api()
        self.llm = api.get_llm()

    def _parse_scores(self, text: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[str], str]:
        """Parse JSON scores first; fallback to SCORE only if needed."""
        # First try JSON (including fenced or embedded).
        candidates: List[Tuple[str, str]] = []
        for match in _JSON_FENCE_RE.finditer(text):
            candidates.append((match.group(1), "json_fenced"))
        obj_match = _JSON_OBJ_RE.search(text)
        if obj_match:
            candidates.append((obj_match.group(0), "json_inline"))
        candidates.append((text.strip(), "json"))

        seen = set()
        for cand, mode in candidates:
            if not cand or cand in seen:
                continue
            seen.add(cand)
            try:
                data = json.loads(cand)
            except Exception:
                continue
            if isinstance(data, dict):
                clarity = data.get("clarity")
                cogency = data.get("cogency")
                relevance = data.get("relevance")
                overall = data.get("overall")
                rationale = data.get("rationale")

                clarity_v = _clip01(float(clarity)) if clarity is not None else None
                cogency_v = _clip01(float(cogency)) if cogency is not None else None
                relevance_v = _clip01(float(relevance)) if relevance is not None else None

                if overall is not None:
                    overall_v = _clip01(float(overall))
                elif None not in (clarity_v, cogency_v, relevance_v):
                    overall_v = _clip01(
                        clarity_v * self.weights["clarity"]
                        + cogency_v * self.weights["cogency"]
                        + relevance_v * self.weights["relevance"]
                    )
                else:
                    overall_v = None

                if any(v is not None for v in (clarity_v, cogency_v, relevance_v, overall_v)):
                    return clarity_v, cogency_v, relevance_v, overall_v, rationale, mode

        # Fallback: SCORE only (overall), do not replicate into sub-scores.
        match = _SCORE_RE.search(text)
        if match:
            overall_v = _clip01(float(match.group(1)))
            return None, None, None, overall_v, None, "score"

        # Fallback: first float as overall only.
        vals = [float(v) for v in _FLOAT_RE.findall(text)[:1]]
        if len(vals) == 1:
            overall_v = _clip01(vals[0])
            return None, None, None, overall_v, None, "float"

        return None, None, None, None, None, "none"

    def score_argument(self, text: str, context: str = "") -> Tuple[JudgeScore, str, str, Optional[str]]:
        # Truncate inputs to avoid token-limit failures on verbose KG responses.
        text_trunc = text[:2500].rstrip()
        context_trunc = context[:1200].rstrip()

        prompt = (
            "You are an expert annotator for argument quality in the style of Webis-ArgQuality-20.\n"
            "Return ONLY a JSON object with numeric fields in [0.0, 1.0].\n"
            "Do not include any extra text, markdown, or code fences.\n"
            "Required keys: clarity, cogency, relevance, overall.\n"
            "Optional key: rationale (short string).\n"
            "Example:\n"
            "{\"clarity\":0.72,\"cogency\":0.61,\"relevance\":0.83,\"overall\":0.73,\"rationale\":\"clear and on-topic\"}\n\n"
            f"Argument:\n{text_trunc}\n\n"
            f"Context (opponent's last message):\n{context_trunc}\n"
        )

        response = invoke_with_retry(self.llm.invoke, prompt, operation="judge.invoke")
        raw = response.content if hasattr(response, "content") else str(response)
        clarity, cogency, relevance, overall, rationale, parse_mode = self._parse_scores(raw)
        if overall is None and None not in (clarity, cogency, relevance):
            overall = _clip01(
                clarity * self.weights["clarity"]
                + cogency * self.weights["cogency"]
                + relevance * self.weights["relevance"]
            )
        if clarity is None and cogency is None and relevance is None:
            warnings.warn(
                f"[judge] _parse_scores returned no sub-scores. "
                f"Raw response (first 120 chars): {raw[:120]!r}"
            )

        return JudgeScore(clarity=clarity, cogency=cogency, relevance=relevance, overall=overall), raw, parse_mode, rationale


def _find_last_opponent_message(turn_logs: List[Dict[str, Any]], idx: int, module_id: str) -> str:
    for j in range(idx - 1, -1, -1):
        prev_id = turn_logs[j].get("module_id") or turn_logs[j].get("team_id")
        if prev_id != module_id:
            return turn_logs[j].get("message_content", "")
    return ""


def score_log_file(log_path: Path, judge: WebisAlignedOfflineJudge, raw_log_path: Optional[Path] = None) -> Dict[str, Any]:
    with open(log_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    turn_logs = data.get("turn_logs", [])
    per_turn: List[Dict[str, Any]] = []
    parse_mode_counts: Dict[str, int] = {}
    turns_with_subscores = 0
    turns_with_json = 0

    total_scoring_turns = sum(
        1
        for t in turn_logs
        if t.get("phase") in {"debate", "standoff"} and t.get("message_content")
    )
    scored_turns = 0

    for idx, turn in enumerate(turn_logs):
        if turn.get("phase") not in {"debate", "standoff"}:
            continue
        message = turn.get("message_content", "")
        if not message:
            continue

        module_id = turn.get("module_id") or turn.get("team_id", "")
        context = _find_last_opponent_message(turn_logs, idx, module_id)
        score, raw, parse_mode, rationale = judge.score_argument(message, context)
        scored_turns += 1
        if scored_turns == 1 or scored_turns % 5 == 0 or scored_turns == total_scoring_turns:
            print(
                f"[judge] {log_path.stem}: {scored_turns}/{total_scoring_turns} turns scored",
                flush=True,
            )
        has_subscores = all(v is not None for v in (score.clarity, score.cogency, score.relevance))
        parse_mode_counts[parse_mode] = parse_mode_counts.get(parse_mode, 0) + 1
        if parse_mode.startswith("json"):
            turns_with_json += 1
        if has_subscores:
            turns_with_subscores += 1

        if raw_log_path is not None:
            record = {
                "session_id": data.get("session_info", {}).get("session_id"),
                "turn_number": turn.get("turn_number"),
                "module_id": module_id,
                "clarity": score.clarity,
                "cogency": score.cogency,
                "relevance": score.relevance,
                "overall": score.overall,
                "parse_mode": parse_mode,
                "has_subscores": has_subscores,
                "rationale": rationale,
                "raw_output": raw,
            }
            with raw_log_path.open("a", encoding="utf-8") as rf:
                rf.write(json.dumps(record, ensure_ascii=True) + "\n")

        per_turn.append({
            "turn_number": turn.get("turn_number"),
            "module_id": module_id,
            "module_label": turn.get("module_label"),
            "agent_id": turn.get("agent_id") or turn.get("module_id"),
            "clarity": score.clarity,
            "cogency": score.cogency,
            "relevance": score.relevance,
            "overall": score.overall,
            "parse_mode": parse_mode,
            "has_subscores": has_subscores,
        })

    if per_turn:
        def avg(key: str) -> Optional[float]:
            vals = [t[key] for t in per_turn if t.get(key) is not None]
            return sum(vals) / len(vals) if vals else None
        summary = {
            "turns_scored": len(per_turn),
            "turns_with_subscores": turns_with_subscores,
            "turns_with_json": turns_with_json,
            "parse_mode_counts": parse_mode_counts,
            "avg_clarity": avg("clarity"),
            "avg_cogency": avg("cogency"),
            "avg_relevance": avg("relevance"),
            "avg_overall": avg("overall"),
        }
        json_ratio = turns_with_json / len(per_turn)
        subs_ratio = turns_with_subscores / len(per_turn)
        if json_ratio < MIN_JSON_RATIO_WARN:
            warnings.warn(
                f"[judge] Low JSON parse rate: {json_ratio:.2%} (warn<{MIN_JSON_RATIO_WARN:.0%})."
            )
        if subs_ratio < MIN_SUBSCORE_RATIO_WARN:
            warnings.warn(
                f"[judge] Low sub-score coverage: {subs_ratio:.2%} (warn<{MIN_SUBSCORE_RATIO_WARN:.0%})."
            )
        if json_ratio < MIN_JSON_RATIO_FAIL:
            raise RuntimeError(
                f"[judge] JSON parse rate below fail threshold: {json_ratio:.2%} (<{MIN_JSON_RATIO_FAIL:.0%})."
            )
        if subs_ratio < MIN_SUBSCORE_RATIO_FAIL:
            raise RuntimeError(
                f"[judge] Sub-score coverage below fail threshold: {subs_ratio:.2%} (<{MIN_SUBSCORE_RATIO_FAIL:.0%})."
            )
    else:
        summary = {
            "turns_scored": 0,
            "turns_with_subscores": 0,
            "turns_with_json": 0,
            "parse_mode_counts": {},
            "avg_clarity": None,
            "avg_cogency": None,
            "avg_relevance": None,
            "avg_overall": None,
        }

    return {
        "log_path": str(log_path),
        "session_id": data.get("session_info", {}).get("session_id"),
        "summary": summary,
        "per_turn": per_turn,
    }


def _load_log_paths(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.json") if p.is_file())


def _write_csv(output_csv: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def check_e1_passed(status_path: Path) -> Tuple[bool, str]:
    if not status_path.exists():
        return False, "E1 status file not found"

    with open(status_path, "r", encoding="utf-8") as f:
        status = json.load(f)

    # Reject any status file generated with the --force-pass bypass (dev artifact).
    if status.get("force_pass") is True:
        return False, (
            "E1 status was written with force_pass=True (dev bypass). "
            "Re-run run_e1_judge_calibration.py without --force-pass on a real dataset."
        )

    if status.get("passed") is True:
        return True, "E1 passed"
    return False, status.get("reason", "E1 did not pass")


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline Webis-aligned judge for debate logs")
    parser.add_argument("--logs", required=True, help="Path to a log JSON file or a directory")
    parser.add_argument("--output-dir", default="analysis/outputs/judge", help="Output directory")
    parser.add_argument("--raw-log", default=None, help="Optional JSONL path to store raw judge outputs")
    parser.add_argument("--require-e1", action="store_true", help="Require E1 gate before scoring")
    parser.add_argument("--e1-status", default="analysis/judge_e1/status.json", help="E1 status file path")
    args = parser.parse_args()

    if args.require_e1:
        ok, reason = check_e1_passed(Path(args.e1_status))
        if not ok:
            print(f"E1 gate failed: {reason}")
            return 2

    judge = WebisAlignedOfflineJudge()
    logs_path = Path(args.logs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    raw_log_path = Path(args.raw_log) if args.raw_log else None
    if raw_log_path:
        raw_log_path.parent.mkdir(parents=True, exist_ok=True)

    for log_path in _load_log_paths(logs_path):
        result = score_log_file(log_path, judge, raw_log_path=raw_log_path)
        session_id = result.get("session_id") or log_path.stem

        out_json = output_dir / f"judge_{session_id}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        for row in result["per_turn"]:
            row = dict(row)
            row["session_id"] = session_id
            all_rows.append(row)

        summary = dict(result["summary"])
        summary["session_id"] = session_id
        summaries.append(summary)

    _write_csv(output_dir / "judge_scores_turns.csv", all_rows)
    _write_csv(output_dir / "judge_scores_summary.csv", summaries)

    print(f"Saved judge outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
