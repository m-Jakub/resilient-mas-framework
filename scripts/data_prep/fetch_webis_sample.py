"""
Utility: prepare a small Webis-ArgQuality-20 compatible sample dataset.

This script does NOT download copyrighted data. It creates a mock 50-item
sample with the same schema expected by run_e1_judge_calibration.py.

Output fields:
- text
- context
- label (0.0-1.0)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List


ARGUMENT_TEMPLATES = [
    "Because {principle}, we should {action}, even if {tradeoff}.",
    "Given that {fact}, it follows that {action} is the most defensible option.",
    "Since {principle} implies {implication}, rejecting {action} would be inconsistent.",
    "If we accept {principle}, then {action} is required despite {tradeoff}.",
    "The strongest reason for {action} is {fact}, which outweighs {tradeoff}.",
]

CONTEXT_TEMPLATES = [
    "Opponent argues that {counter} and prioritizes {principle} above all else.",
    "Previous claim: {counter}. The debate centers on {principle}.",
    "Context: the critic insists on {counter}, citing {fact}.",
]

PRINCIPLES = [
    "respecting autonomy",
    "minimizing harm",
    "maximizing welfare",
    "upholding duty",
    "protecting rights",
]

ACTIONS = [
    "allocate resources to the most vulnerable",
    "reject coercive measures",
    "prioritize lifesaving interventions",
    "avoid sacrificial tradeoffs",
    "enforce a strict fairness rule",
]

TRADEOFFS = [
    "some individuals may be worse off",
    "the outcome may feel unfair",
    "short-term costs will increase",
    "certain cases will be excluded",
    "public approval may decline",
]

FACTS = [
    "the evidence is incomplete",
    "the risk distribution is skewed",
    "there is limited capacity",
    "the situation is time-critical",
    "the long-term effects are uncertain",
]

COUNTERS = [
    "outcomes matter more than intentions",
    "rules should never be broken",
    "individual rights override aggregate welfare",
    "exceptions undermine the rule",
    "procedural fairness is paramount",
]


def _make_item(rng: random.Random) -> Dict[str, str | float]:
    principle = rng.choice(PRINCIPLES)
    action = rng.choice(ACTIONS)
    tradeoff = rng.choice(TRADEOFFS)
    fact = rng.choice(FACTS)
    counter = rng.choice(COUNTERS)

    text = rng.choice(ARGUMENT_TEMPLATES).format(
        principle=principle,
        action=action,
        tradeoff=tradeoff,
        fact=fact,
        implication=action,
    )
    context = rng.choice(CONTEXT_TEMPLATES).format(
        counter=counter,
        principle=principle,
        fact=fact,
    )

    # Mock gold label: coherent arguments with explicit reasoning get higher scores.
    label = rng.uniform(0.35, 0.9)
    return {"text": text, "context": context, "label": round(label, 3)}


def _write_json(path: Path, items: List[Dict[str, str | float]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"items": items}, f, indent=2, ensure_ascii=False)


def _write_csv(path: Path, items: List[Dict[str, str | float]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "context", "label"])
        writer.writeheader()
        for item in items:
            writer.writerow(item)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a mock Webis-ArgQuality-20 sample")
    parser.add_argument("--count", type=int, default=50, help="Number of items to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", default="data/webis_argquality_sample.json", help="Output file path")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    items = [_make_item(rng) for _ in range(args.count)]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".csv":
        _write_csv(out_path, items)
    else:
        _write_json(out_path, items)

    print(f"Saved mock dataset to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
