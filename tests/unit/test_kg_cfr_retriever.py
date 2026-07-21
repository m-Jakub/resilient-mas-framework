"""
Regression test: kg_cfr_full must wire a non-None rag_retriever into cr_engine.run().

Bug: After condition rename (commit a0c9ecb), _normalize_cfr_mode() stored
     "kg_cfr_full" but _counterfactual_reflection() compared against the legacy
     string "knowledge-grounded". Branch was always False → rag_retriever stayed
     None → kg_cfr_full behaved identically to cfr_no_kg in all existing runs.

Fix: line 196 of aegis_orchestrator.py: "knowledge-grounded" → "kg_cfr_full"
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure repo root + src/ are importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from src.kg_cfr.aegis_orchestrator import AegisState, TripartiteStandoff


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_agent() -> MagicMock:
    """Minimal agent with a populated identity_graph."""
    agent = MagicMock()
    agent.identity_graph = MagicMock()
    agent.identity_graph.get_beliefs_about_concept.return_value = [
        {"content": "Individual rights are inviolable."},
        {"content": "Justice demands proportionality."},
    ]
    agent.identity_graph.get_core_beliefs.return_value = [
        {"content": "Core belief: act on principle."},
    ]
    return agent


def _make_minimal_state() -> AegisState:
    """AegisState with all required keys populated."""
    mock_module = MagicMock()
    mock_module.agent = _make_mock_agent()

    return {
        "modules": {"shm": mock_module},
        "module_order": ["shm"],
        "chat_history": [],
        "current_module_id": "shm",
        "turn_count": 2,
        "crisis_context": "Power grid allocation crisis",
        "shock_event": None,
        "module_labels": {"shm": "SHM"},
        "private_thought_buffer": {},
        "cfr_mode": "kg_cfr_full",   # field not used by orchestrator internals
        "cf_flags": {},
        "use_id_rag": True,
        "use_adaptive_idrag": True,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestKgCfrRetriever:
    """Guard against the 'knowledge-grounded' vs 'kg_cfr_full' string mismatch."""

    def test_kg_cfr_full_passes_nonnull_rag_retriever(self):
        """
        kg_cfr_full must pass a **non-None** rag_retriever to cr_engine.run().
        If this assertion fails, KG evidence will never be retrieved, and
        kg_cfr_full is silently identical to cfr_no_kg.
        """
        standoff = TripartiteStandoff(cfr_mode="kg_cfr_full", enable_logging=False)
        standoff.cr_engine = MagicMock()
        standoff.cr_engine.run.return_value = None  # don't need a real result
        standoff.context_builder = MagicMock()
        standoff.context_builder.build_attack_vector_with_claims.return_value = (
            "Opponent argues utility outweighs individual rights.",
            [],
        )

        state = _make_minimal_state()
        standoff._counterfactual_reflection(state)

        assert standoff.cr_engine.run.called, (
            "cr_engine.run() was not called at all — CFR was unexpectedly skipped."
        )
        call_kwargs = standoff.cr_engine.run.call_args.kwargs
        rag_retriever = call_kwargs.get("rag_retriever")

        assert rag_retriever is not None, (
            "kg_cfr_full must pass a non-None rag_retriever to cr_engine.run(). "
            "Got None — regression: cfr_mode string is likely still 'knowledge-grounded'."
        )

    def test_kg_cfr_full_rag_retriever_returns_list(self):
        """
        The rag_retriever closure for kg_cfr_full must return a list of strings
        when called with a query.
        """
        standoff = TripartiteStandoff(cfr_mode="kg_cfr_full", enable_logging=False)
        standoff.cr_engine = MagicMock()
        standoff.cr_engine.run.return_value = None
        standoff.context_builder = MagicMock()
        standoff.context_builder.build_attack_vector_with_claims.return_value = (
            "Rights vs utility framing.",
            [],
        )

        state = _make_minimal_state()
        standoff._counterfactual_reflection(state)

        call_kwargs = standoff.cr_engine.run.call_args.kwargs
        rag_retriever = call_kwargs["rag_retriever"]

        result = rag_retriever("duty justice autonomy")
        assert isinstance(result, list), (
            f"rag_retriever must return list, got {type(result).__name__}"
        )
        assert all(isinstance(s, str) for s in result), (
            "rag_retriever must return a list of strings."
        )

    def test_kg_cfr_full_rag_retriever_nonempty_for_matching_keywords(self):
        """
        When the mock identity_graph returns beliefs, the rag_retriever must
        yield at least one evidence string.
        """
        standoff = TripartiteStandoff(cfr_mode="kg_cfr_full", enable_logging=False)
        standoff.cr_engine = MagicMock()
        standoff.cr_engine.run.return_value = None
        standoff.context_builder = MagicMock()
        standoff.context_builder.build_attack_vector_with_claims.return_value = (
            "duty justice rights",
            [],
        )

        state = _make_minimal_state()
        # identity_graph returns known beliefs
        state["modules"]["shm"].agent.identity_graph.get_beliefs_about_concept.return_value = [
            {"content": "Rights are non-negotiable."}
        ]

        standoff._counterfactual_reflection(state)

        rag_retriever = standoff.cr_engine.run.call_args.kwargs["rag_retriever"]
        result = rag_retriever("rights justice")
        assert len(result) > 0, (
            "rag_retriever should return at least one belief string for matching keywords."
        )

    def test_cfr_no_kg_passes_null_rag_retriever(self):
        """
        cfr_no_kg must pass rag_retriever=None — no doctrinal evidence retrieval.
        Verifies the KG path stays inactive for the correct condition.
        """
        standoff = TripartiteStandoff(cfr_mode="cfr_no_kg", enable_logging=False)
        standoff.cr_engine = MagicMock()
        standoff.cr_engine.run.return_value = None
        standoff.context_builder = MagicMock()
        standoff.context_builder.build_attack_vector_with_claims.return_value = (
            "Opponent argues security over freedom.",
            [],
        )

        state = _make_minimal_state()
        standoff._counterfactual_reflection(state)

        assert standoff.cr_engine.run.called, (
            "cr_engine.run() was not called at all for cfr_no_kg."
        )
        call_kwargs = standoff.cr_engine.run.call_args.kwargs
        rag_retriever = call_kwargs.get("rag_retriever")

        assert rag_retriever is None, (
            f"cfr_no_kg must pass rag_retriever=None to cr_engine.run(). "
            f"Got non-None: {rag_retriever!r}"
        )

    def test_cfr_mode_normalization_accepts_legacy_alias(self):
        """
        Passing the legacy alias 'knowledge-grounded' must produce the same
        canonical cfr_mode as 'kg_cfr_full', and still wire a non-None retriever.
        """
        standoff = TripartiteStandoff(cfr_mode="knowledge-grounded", enable_logging=False)
        assert standoff.cfr_mode == "kg_cfr_full", (
            f"_normalize_cfr_mode should map 'knowledge-grounded' → 'kg_cfr_full', "
            f"got: '{standoff.cfr_mode}'"
        )

        standoff.cr_engine = MagicMock()
        standoff.cr_engine.run.return_value = None
        standoff.context_builder = MagicMock()
        standoff.context_builder.build_attack_vector_with_claims.return_value = ("x", [])

        state = _make_minimal_state()
        standoff._counterfactual_reflection(state)

        call_kwargs = standoff.cr_engine.run.call_args.kwargs
        assert call_kwargs.get("rag_retriever") is not None, (
            "Legacy alias 'knowledge-grounded' must produce a non-None rag_retriever."
        )

    def test_no_cfr_baseline_skips_cr_engine_entirely(self):
        """
        no_cfr_baseline must not call cr_engine.run() at all.
        (cr_engine is None for this condition by design.)
        """
        standoff = TripartiteStandoff(cfr_mode="no_cfr_baseline", enable_logging=False)
        assert standoff.cr_engine is None, (
            "no_cfr_baseline should have cr_engine=None."
        )

        state = _make_minimal_state()
        result = standoff._counterfactual_reflection(state)

        assert result["cf_flags"].get("activated") is False, (
            "no_cfr_baseline must set cf_flags['activated']=False."
        )
