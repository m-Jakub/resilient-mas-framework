"""
Estimate token usage and cost from debate logs.

This script reads conversation logger JSON logs (logs/production or logs/experiments)
and estimates token usage using a simple chars-per-token heuristic.

It then estimates cost using provided price per 1M tokens.
"""

from __future__ import annotations

import argparse
import json
import glob
from pathlib import Path
from typing import Iterable, Dict, Any, List, Tuple

DEFAULT_CHARS_PER_TOKEN = 4.0  # rough heuristic
DEFAULT_INPUT_MULTIPLIER = 1.2  # prompt tokens ~= output_tokens * multiplier


def _iter_paths(patterns: List[str]) -> Iterable[Path]:
    seen = set()
    for pat in patterns:
        for p in glob.glob(pat):
            path = Path(p)
            if path.is_file() and path not in seen:
                seen.add(path)
                yield path


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _extract_text_fields(data: Dict[str, Any]) -> List[str]:
    texts: List[str] = []

    # Turn logs (main outputs)
    for turn in data.get("turn_logs", []):
        content = turn.get("message_content")
        if content:
            texts.append(content)

    # Moderator questions (also LLM outputs in many runs)
    for mod in data.get("moderation_logs", []):
        question = mod.get("question")
        if question:
            texts.append(question)
        team_response = mod.get("team_response")
        if team_response:
            texts.append(team_response)

    # Ontology analysis outputs (short; include optionally via flag in future)
    return texts


def _extract_usage_metadata(data: Dict[str, Any]) -> Tuple[int, int]:
    """
    Extract token usage from usage_metadata in turn_logs.

    Returns:
        (input_tokens, output_tokens)
    """
    input_tokens = 0
    output_tokens = 0

    for turn in data.get("turn_logs", []):
        meta = turn.get("metadata") or {}
        usage = meta.get("usage_metadata") if isinstance(meta, dict) else None
        if not isinstance(usage, dict):
            continue
        input_tokens += int(usage.get("input_tokens", 0) or 0)
        output_tokens += int(usage.get("output_tokens", 0) or 0)

    return input_tokens, output_tokens


def _estimate_tokens(texts: List[str], chars_per_token: float) -> int:
    total_chars = sum(len(t) for t in texts)
    return int(round(total_chars / chars_per_token))


def _format_money(value: float) -> str:
    return f"{value:.4f}"


def estimate_file(
    path: Path,
    chars_per_token: float,
    input_multiplier: float,
    price_input_per_million: float,
    price_output_per_million: float,
) -> Dict[str, Any]:
    data = _load_json(path)
    usage_input, usage_output = _extract_usage_metadata(data)

    if usage_input > 0 or usage_output > 0:
        input_tokens = usage_input
        output_tokens = usage_output
        estimate_mode = "usage_metadata"
    else:
        texts = _extract_text_fields(data)
        output_tokens = _estimate_tokens(texts, chars_per_token)
        input_tokens = int(round(output_tokens * input_multiplier))
        estimate_mode = "heuristic"

    cost_output = (output_tokens / 1_000_000.0) * price_output_per_million
    cost_input = (input_tokens / 1_000_000.0) * price_input_per_million

    return {
        "file": str(path),
        "output_tokens": output_tokens,
        "input_tokens": input_tokens,
        "cost_output": cost_output,
        "cost_input": cost_input,
        "cost_total": cost_output + cost_input,
        "estimate_mode": estimate_mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate token usage and cost from debate logs."
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=["logs/production/*.json", "logs/experiments/*.json"],
        help="Glob patterns for log files to estimate.",
    )
    parser.add_argument(
        "--chars-per-token",
        type=float,
        default=DEFAULT_CHARS_PER_TOKEN,
        help=f"Chars-per-token heuristic (default: {DEFAULT_CHARS_PER_TOKEN}).",
    )
    parser.add_argument(
        "--input-multiplier",
        type=float,
        default=DEFAULT_INPUT_MULTIPLIER,
        help=f"Estimate input tokens as output_tokens * multiplier (default: {DEFAULT_INPUT_MULTIPLIER}).",
    )
    parser.add_argument(
        "--price-input",
        type=float,
        required=True,
        help="Price per 1M input tokens (USD). Example: 0.35",
    )
    parser.add_argument(
        "--price-output",
        type=float,
        required=True,
        help="Price per 1M output tokens (USD). Example: 0.70",
    )

    args = parser.parse_args()

    results = []
    for path in _iter_paths(args.paths):
        results.append(
            estimate_file(
                path,
                chars_per_token=args.chars_per_token,
                input_multiplier=args.input_multiplier,
                price_input_per_million=args.price_input,
                price_output_per_million=args.price_output,
            )
        )

    if not results:
        print("No log files matched.")
        return

    total_output_tokens = sum(r["output_tokens"] for r in results)
    total_input_tokens = sum(r["input_tokens"] for r in results)
    total_cost_output = sum(r["cost_output"] for r in results)
    total_cost_input = sum(r["cost_input"] for r in results)

    print("\n=== COST ESTIMATE SUMMARY ===")
    for r in results:
        print(
            f"- {r['file']}: output_tokens={r['output_tokens']}, "
            f"input_tokens={r['input_tokens']}, "
            f"cost=${_format_money(r['cost_total'])} "
            f"(mode={r['estimate_mode']})"
        )

    print("\n=== TOTAL ===")
    print(f"Output tokens: {total_output_tokens}")
    print(f"Input tokens (estimated): {total_input_tokens}")
    print(f"Cost output: ${_format_money(total_cost_output)}")
    print(f"Cost input: ${_format_money(total_cost_input)}")
    print(f"Cost total: ${_format_money(total_cost_output + total_cost_input)}")

    print("\nNOTE: Input tokens are estimated from outputs via multiplier.")
    print("      Adjust --input-multiplier and --chars-per-token as needed.")


if __name__ == "__main__":
    main()
