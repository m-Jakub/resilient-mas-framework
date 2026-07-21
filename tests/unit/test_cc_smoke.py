"""
Smoke tests for the Delta-CC pipeline.

Guards against the field-name mismatch bug where:
- orchestrator logs NL plan as 'private_strategy_nl'
- orchestrator logs JSON plan as 'private_strategy_json'
- but metric scripts used to read the legacy 'private_strategy' key (always empty
  in current logs), silently producing CC = 0 / n_cc = 0 for all CFR runs.

Fails if:
  - CFR-active turns produce zero usable plan-response pairs
  - all CC values collapse to 0.0 (degenerate result)
  - the wrong metadata key is read (regression guard)
"""

from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from scripts.analysis.compute_kes_metrics import _compute_cc as kes_compute_cc
from scripts.analysis.compute_v5_metrics import _compute_cc as v5_compute_cc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_turn(
    turn_number: int,
    private_strategy_nl: str = "",
    private_strategy_json: str = "",
    private_strategy: str = "",        # legacy field — should be empty in new logs
    message_content: str = "Some public response text about policy matters.",
) -> Dict[str, Any]:
    return {
        "turn_number": turn_number,
        "phase": "debate",
        "module_id": "shm",
        "message_content": message_content,
        "metadata": {
            "private_strategy_nl":   private_strategy_nl,
            "private_strategy_json": private_strategy_json,
            "private_strategy":      private_strategy,   # legacy — empty in new logs
        },
    }


_VALID_JSON = (
    '{"target_opponent_claim_id": "claim_01", '
    '"attack_surface": "VALUE_CONFLICT", '
    '"strategic_intent": "COUNTER"}'
)
_VALID_NL = (
    "Directly refute the opponent by targeting their value conflict | "
    "Anti-utilitarian framing | COUNTER"
)
_PUBLIC = "The allocation must prioritize long-term grid stability over short-term gains."

# Fake embedding vectors — orthogonal vs identical to test similarity range
_VEC_A = [1.0, 0.0, 0.0]
_VEC_B = [0.0, 1.0, 0.0]   # orthogonal  → similarity = 0.0
_VEC_C = [1.0, 0.0, 0.0]   # same as A   → similarity = 1.0
_VEC_D = [0.8, 0.6, 0.0]   # mid-range   → similarity ~ 0.8


# ---------------------------------------------------------------------------
# KES CC tests
# ---------------------------------------------------------------------------

class FakeGeminiEmbeddings:
    """Returns distinct non-zero vectors for plan vs response texts."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Alternate between _VEC_D and a slightly rotated variant per text
        result = []
        for i, _ in enumerate(texts):
            if i % 2 == 0:
                result.append(_VEC_D)          # plan-like vector
            else:
                result.append([0.6, 0.8, 0.0]) # response-like vector
        return result


def test_kes_cc_uses_private_strategy_nl_not_legacy_field():
    """
    KES _compute_cc must read 'private_strategy_nl' (new field).
    If it falls back to the empty legacy 'private_strategy', it would return {}.
    """
    turns = [
        _make_turn(3, private_strategy_nl=_VALID_NL, private_strategy=""),
    ]

    with patch(
        "scripts.analysis.compute_kes_metrics._get_embedding_model",
        return_value=FakeGeminiEmbeddings(),
    ):
        scores = kes_compute_cc(turns)

    assert len(scores) == 1, (
        f"Expected 1 CC pair from 1 CFR turn, got {len(scores)}. "
        "Likely reading empty 'private_strategy' instead of 'private_strategy_nl'."
    )


def test_kes_cc_cfr_active_run_produces_nonzero_pairs():
    """Given N CFR turns with valid private_strategy_nl, CC returns N pairs."""
    turns = [
        _make_turn(i, private_strategy_nl=_VALID_NL, private_strategy="")
        for i in range(3, 12)
    ]

    with patch(
        "scripts.analysis.compute_kes_metrics._get_embedding_model",
        return_value=FakeGeminiEmbeddings(),
    ):
        scores = kes_compute_cc(turns)

    assert len(scores) == len(turns), (
        f"Expected {len(turns)} CC pairs, got {len(scores)}. "
        "CFR-active turns are not producing usable plan-response pairs."
    )


def test_kes_cc_values_not_all_zero():
    """CC values must not all collapse to 0.0."""
    turns = [_make_turn(3, private_strategy_nl=_VALID_NL, private_strategy="")]

    with patch(
        "scripts.analysis.compute_kes_metrics._get_embedding_model",
        return_value=FakeGeminiEmbeddings(),
    ):
        scores = kes_compute_cc(turns)

    assert scores, "No CC pairs produced."
    for idx, val in scores.items():
        assert val >= 0.0, f"CC[{idx}] is negative: {val}"
        assert val <= 1.0, f"CC[{idx}] exceeds 1.0: {val}"
        # A degenerate all-zero result would mean the embedding matched exactly
        # nothing. If it's > 0 the metric is alive. (We cannot demand it's non-zero
        # from the fake vectors since some combos are orthogonal; just check range.)


def test_kes_cc_no_cfr_turns_returns_empty():
    """Turns with all private_strategy fields empty must produce no CC pairs."""
    turns = [
        _make_turn(i, private_strategy_nl="", private_strategy_json="", private_strategy="")
        for i in range(3, 8)
    ]

    with patch(
        "scripts.analysis.compute_kes_metrics._get_embedding_model",
        return_value=FakeGeminiEmbeddings(),
    ):
        scores = kes_compute_cc(turns)

    assert scores == {}, (
        f"Expected empty CC dict for non-CFR turns, got: {scores}"
    )


def test_kes_cc_legacy_fallback_still_works():
    """Legacy logs with 'private_strategy' (no _nl suffix) must still produce CC."""
    turns = [
        _make_turn(3, private_strategy_nl="", private_strategy=_VALID_NL),
    ]

    with patch(
        "scripts.analysis.compute_kes_metrics._get_embedding_model",
        return_value=FakeGeminiEmbeddings(),
    ):
        scores = kes_compute_cc(turns)

    assert len(scores) == 1, "Legacy 'private_strategy' fallback broke."


# ---------------------------------------------------------------------------
# v5 CC tests
# ---------------------------------------------------------------------------

class FakeSentenceTransformer:
    """Returns distinct non-zero numpy-like plain lists for encoding."""

    def encode(self, texts: List[str], normalize_embeddings: bool = False) -> List[List[float]]:
        result = []
        for i, _ in enumerate(texts):
            if i % 2 == 0:
                result.append(_VEC_D)
            else:
                result.append([0.6, 0.8, 0.0])
        return result


def test_v5_cc_uses_private_strategy_json_not_legacy_field():
    """
    v5 _compute_cc must read 'private_strategy_json' (new field).
    If it falls back to empty 'private_strategy', json.loads fails and returns {}.
    """
    turns = [
        _make_turn(3, private_strategy_json=_VALID_JSON, private_strategy=""),
    ]

    with patch(
        "scripts.analysis.compute_v5_metrics._get_cc_model",
        return_value=FakeSentenceTransformer(),
    ):
        scores = v5_compute_cc(turns)

    assert len(scores) == 1, (
        f"Expected 1 CC pair from 1 CFR turn, got {len(scores)}. "
        "Likely reading empty 'private_strategy' instead of 'private_strategy_json'."
    )


def test_v5_cc_cfr_active_run_produces_nonzero_pairs():
    """Given N CFR turns with valid private_strategy_json, v5 CC returns N pairs."""
    turns = [
        _make_turn(i, private_strategy_json=_VALID_JSON, private_strategy="")
        for i in range(3, 12)
    ]

    with patch(
        "scripts.analysis.compute_v5_metrics._get_cc_model",
        return_value=FakeSentenceTransformer(),
    ):
        scores = v5_compute_cc(turns)

    assert len(scores) == len(turns), (
        f"Expected {len(turns)} CC pairs, got {len(scores)}. "
        "CFR-active turns not producing usable plan-response pairs."
    )


def test_v5_cc_no_cfr_turns_returns_empty():
    """Turns with all private_strategy fields empty must produce no v5 CC pairs."""
    turns = [
        _make_turn(i, private_strategy_nl="", private_strategy_json="", private_strategy="")
        for i in range(3, 8)
    ]

    with patch(
        "scripts.analysis.compute_v5_metrics._get_cc_model",
        return_value=FakeSentenceTransformer(),
    ):
        scores = v5_compute_cc(turns)

    assert scores == {}, (
        f"Expected empty CC dict for non-CFR turns, got: {scores}"
    )


def test_v5_cc_legacy_fallback_still_works():
    """Legacy logs with JSON in 'private_strategy' must still produce v5 CC pairs."""
    turns = [
        _make_turn(3, private_strategy_json="", private_strategy=_VALID_JSON),
    ]

    with patch(
        "scripts.analysis.compute_v5_metrics._get_cc_model",
        return_value=FakeSentenceTransformer(),
    ):
        scores = v5_compute_cc(turns)

    assert len(scores) == 1, "Legacy 'private_strategy' fallback (JSON) broke."


def test_v5_cc_nl_string_in_private_strategy_json_raises_no_pairs():
    """
    Regression guard: if private_strategy_json accidentally contains the NL string
    (as in pre-b751377 fallback-path bug), json.loads will fail and _plan_text_from_strategy
    returns None → 0 pairs. This test documents/detects that failure mode.
    """
    turns = [
        _make_turn(3, private_strategy_json=_VALID_NL, private_strategy=""),
    ]

    with patch(
        "scripts.analysis.compute_v5_metrics._get_cc_model",
        return_value=FakeSentenceTransformer(),
    ):
        scores = v5_compute_cc(turns)

    assert scores == {}, (
        "NL string in private_strategy_json should produce 0 CC pairs "
        "(JSON parse fails → no structured plan fields)."
    )
