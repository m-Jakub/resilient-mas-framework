"""
Integration test: CFR privacy and logging.
Ensures private_thought_buffer is not leaked to chat_history or logs.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

# Add repository root and src to path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from langchain.schema import HumanMessage
from src.hde.team import Team
import src.hde.team_conversation as tc


class DummyLLM:
    def invoke(self, *args, **kwargs):
        return SimpleNamespace(content="dummy")


class DummyAPI:
    def get_llm(self):
        return DummyLLM()

    def get_model_name(self) -> str:
        return "dummy"


class MockAgent:
    def __init__(self, key: str, name: str):
        self.key = key
        self.name = name
        self.school = key
        self.persona = f"Mock {name}"
        self.last_private_strategy = None

    def respond(
        self,
        topic: str,
        chat_history: list,
        ontology_insights: list = None,
        use_id_rag: bool = True,
        use_adaptive_idrag: bool = True,
        private_strategy: str = "",
    ):
        self.last_private_strategy = private_strategy
        return "Public response without private data."


def test_private_strategy_not_leaked(monkeypatch):
    monkeypatch.setattr(tc, "get_philosopher_api", lambda: DummyAPI())

    conversation = tc.TeamPhilosophicalConversation(
        deliberation_rounds=1,
        debate_turns=6,
        enable_moderation=False,
        enable_logging=True,
        session_id="test_cr_privacy_001",
        use_tom=False,
        enable_adversarial_perturbations=False,
        cfr_mode="prompt-only",
    )

    team = Team("team_a", "Team A")
    agent = MockAgent("agent1", "Agent One")
    team.add_agent("agent1", agent)

    state = tc.TeamConversationState(
        teams={"team_a": team},
        chat_history=[HumanMessage(content="duty and dignity", name="Opponent")],
        current_speaker_team="team_a",
        current_speaker_agent="agent1",
        current_phase="debate",
        turn_count=3,
        deliberation_count=0,
        ontology_insights=[],
        topic="Test topic",
        moderation_questions=[],
        belief_models={},
        use_id_rag=False,
        use_adaptive_idrag=False,
        perturbation=None,
        private_thought_buffer={},
        cfr_mode="prompt-only",
        cf_flags={},
    )

    state = conversation._counterfactual_reflection(state)
    assert state["cf_flags"]["activated"] is True

    agent_key = "team_a:agent1"
    assert agent_key in state["private_thought_buffer"]
    private_strategy = state["private_thought_buffer"][agent_key]["private_strategy"]

    state = conversation._debate_turn(state)

    # Buffer cleared after use
    assert agent_key not in state.get("private_thought_buffer", {})

    # No private data in chat history (skip check if strategy was empty after guardrails)
    if private_strategy:
        assert all(private_strategy not in msg.content for msg in state["chat_history"])

    # No private data in logs
    logger = conversation.logger
    assert logger is not None
    if private_strategy:
        assert all(private_strategy not in log.message_content for log in logger.turn_logs)
    for log in logger.turn_logs:
        meta_text = str(log.metadata or {})
        assert "private_strategy" not in meta_text

    # CFR metrics recorded
    assert logger.stats.get("cf_activations", 0) >= 1

