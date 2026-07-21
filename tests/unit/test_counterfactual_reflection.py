"""
Unit tests for CounterfactualReflectionEngine.
"""

import sys
from pathlib import Path

# Add repository root and src to path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from src.kg_cfr.counterfactual_reflection import CounterfactualReflectionEngine


def test_should_trigger_keyword_match():
    engine = CounterfactualReflectionEngine()
    should_run, matched, reason = engine.should_trigger(
        turn_idx=3,
        debate_turns=6,
        last_message="This argument rests on duty and dignity.",
        keywords=["duty", "utility"],
        perturbation_active=False,
        random_value=0.99,
    )
    assert should_run is True
    assert reason == "keyword_match"
    assert "duty" in matched


def test_should_trigger_turn_range_block():
    engine = CounterfactualReflectionEngine()
    should_run, _, reason = engine.should_trigger(
        turn_idx=1,
        debate_turns=6,
        last_message="duty",
        keywords=["duty"],
        perturbation_active=False,
        random_value=0.01,
    )
    assert should_run is False
    assert reason == "turn_range"


def test_should_trigger_activation_budget():
    engine = CounterfactualReflectionEngine()
    engine.activations_used = engine.MAX_ACTIVATIONS_PER_DEBATE
    should_run, _, reason = engine.should_trigger(
        turn_idx=3,
        debate_turns=6,
        last_message="duty",
        keywords=["duty"],
        perturbation_active=True,
        random_value=0.01,
    )
    assert should_run is False
    assert reason == "activation_budget"


def test_guardrail_flags_leak_on_banned_phrase():
    engine = CounterfactualReflectionEngine()
    flags = {"identity_ok": True, "leak_risk": False, "doctrinal_drift": False}
    cleaned = engine._apply_guardrails("Internal strategy: do X.", flags)
    assert cleaned == ""
    assert flags["leak_risk"] is True


def test_run_returns_result_when_triggered():
    engine = CounterfactualReflectionEngine()
    result = engine.run(
        topic="Test topic",
        last_message="We must consider duty.",
        turn_idx=3,
        debate_turns=6,
        cfr_mode="prompt-only",
        keywords=["duty"],
        perturbation_active=False,
    )
    assert result is not None
    assert result.guardrail_flags["identity_ok"] is True
    assert "trigger_reason" in result.audit


def test_run_knowledge_grounded_requires_evidence():
    engine = CounterfactualReflectionEngine()

    def retriever(_query: str):
        return ["axiom: never treat persons merely as means"]

    result = engine.run(
        topic="Test topic",
        last_message="We must consider duty.",
        turn_idx=3,
        debate_turns=6,
        cfr_mode="knowledge-grounded",
        keywords=["duty"],
        rag_retriever=retriever,
        perturbation_active=False,
    )
    assert result is not None
    assert result.audit.get("kg_cfr") is True
    assert result.audit.get("rag_docs_count") == 1
