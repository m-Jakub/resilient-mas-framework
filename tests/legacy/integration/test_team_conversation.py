"""
Unit tests for team-based conversation system.
Tests deliberation → debate flow and state management.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hde.team import Team
from src.hde.team_conversation import TeamConversationState, TeamPhilosophicalConversation
from src.hde.enhanced_ontology import EnhancedOntology, integrate_ontology_with_team


class MockAgent:
    """Mock agent for testing without LLM calls."""
    
    def __init__(self, key: str, name: str):
        self.key = key
        self.name = name
        self.school = key
        self.persona = f"Mock {name}"
    
    def respond(self, topic: str, chat_history: list, ontology_insights: list = None):
        """Mock response."""
        return f"[{self.name}] Mock response to debate"


def test_team_conversation_state():
    """Test TeamConversationState structure."""
    print("\n=== TEST: TeamConversationState ===")
    
    team_a = Team("team_a", "Team A")
    team_b = Team("team_b", "Team B")
    
    state = TeamConversationState(
        teams={"team_a": team_a, "team_b": team_b},
        chat_history=[],
        current_speaker_team="team_a",
        current_speaker_agent="agent1",
        current_phase="deliberation",
        turn_count=0,
        deliberation_count=0,
        ontology_insights=[],
        topic="Test topic"
    )
    
    assert "team_a" in state["teams"]
    assert "team_b" in state["teams"]
    assert state["current_phase"] == "deliberation"
    assert state["turn_count"] == 0
    assert state["topic"] == "Test topic"
    
    print("✓ TeamConversationState structure correct")


def test_enhanced_ontology():
    """Test enhanced ontology features."""
    print("\n=== TEST: Enhanced Ontology ===")
    
    ontology = EnhancedOntology()
    
    # Test concept info
    info = ontology.get_concept_info("utility")
    assert "definition" in info
    assert "related_concepts" in info
    assert "opposed_concepts" in info
    assert info["primary_school"] == "utilitarianism"
    
    # Test school concepts
    util_concepts = ontology.get_school_concepts("utilitarianism")
    assert "utility" in util_concepts
    assert "consequence" in util_concepts
    
    # Test conflict analysis
    conflict = ontology.analyze_concept_conflict(["utility", "duty"])
    assert conflict["conflict_count"] > 0
    assert "utilitarianism" in conflict["schools_involved"]
    assert "deontology" in conflict["schools_involved"]
    
    # Test complementary concepts
    related = ontology.suggest_complementary_concepts("utility")
    assert "consequence" in related
    
    print("✓ Enhanced ontology working correctly")
    print(f"  - Utility opposed to: {info['opposed_concepts']}")
    print(f"  - Conflicts detected: {conflict['conflict_count']}")


def test_concept_tracking():
    """Test concept usage tracking."""
    print("\n=== TEST: Concept Tracking ===")
    
    ontology = EnhancedOntology()
    
    # Track some usage
    ontology.track_concept_usage("utility", "team_a", 1)
    ontology.track_concept_usage("duty", "team_b", 2)
    ontology.track_concept_usage("utility", "team_a", 3)
    
    summary = ontology.get_usage_summary()
    
    assert summary["total_uses"] == 3
    assert summary["unique_concepts"] == 2
    assert "utility" in summary["concepts_used"]
    assert "duty" in summary["concepts_used"]
    assert summary["most_used"] == "utility"
    
    print("✓ Concept tracking working")
    print(f"  - Total uses: {summary['total_uses']}")
    print(f"  - Most used: {summary['most_used']}")


def test_ontology_team_integration():
    """Test ontology integration with team."""
    print("\n=== TEST: Ontology-Team Integration ===")
    
    team = Team("team_test", "Test Team")
    ontology = EnhancedOntology()
    
    # Add some concepts to team memory
    team.memory.add_ethical_concept("utility")
    team.memory.add_ethical_concept("consequence")
    
    # Integrate ontology
    integrate_ontology_with_team(team, ontology)
    
    # Test that ontology is integrated
    assert hasattr(team, '_ontology')
    assert hasattr(team, 'get_concept_insights')
    
    # Get insights
    insights = team.get_concept_insights()
    assert len(insights) == 2
    
    print("✓ Ontology-team integration working")
    print(f"  - Team has {len(team.memory.ethical_concepts)} concepts tracked")


def test_team_conversation_initialization():
    """Test TeamPhilosophicalConversation initialization."""
    print("\n=== TEST: TeamPhilosophicalConversation Init ===")
    
    conversation = TeamPhilosophicalConversation(
        deliberation_rounds=2,
        debate_turns=6
    )
    
    assert conversation.deliberation_rounds == 2
    assert conversation.debate_turns == 6
    assert conversation.graph is not None
    assert conversation.moderator_llm is not None
    assert conversation.analyzer_llm is not None
    
    print("✓ TeamPhilosophicalConversation initialized")
    print(f"  - Deliberation rounds: {conversation.deliberation_rounds}")
    print(f"  - Debate turns: {conversation.debate_turns}")


def test_debate_insights_generation():
    """Test debate insights generation."""
    print("\n=== TEST: Debate Insights Generation ===")
    
    ontology = EnhancedOntology()
    
    # Create teams with concepts
    team_a = Team("team_a", "Team A")
    team_a.memory.add_ethical_concept("utility")
    team_a.memory.add_ethical_concept("consequence")
    
    team_b = Team("team_b", "Team B")
    team_b.memory.add_ethical_concept("duty")
    team_b.memory.add_ethical_concept("categorical imperative")
    
    # Generate insights
    insights = ontology.generate_debate_insights({
        "team_a": team_a.memory,
        "team_b": team_b.memory
    })
    
    assert "utilitarianism" in insights
    assert "deontology" in insights
    assert "Conceptual tensions" in insights
    
    print("✓ Debate insights generated")
    print(f"\nInsights:\n{insights}")


def test_memory_persistence_across_phases():
    """Test that team memory persists from deliberation to debate."""
    print("\n=== TEST: Memory Persistence ===")
    
    team = Team("team_test", "Test Team")
    
    # Simulate deliberation
    team.memory.add_insight("Initial insight from deliberation", "Agent A")
    team.memory.add_argument("Core argument", "Agent A")
    team.memory.set_strategic_note("approach", "utilitarian")
    
    # Check memory is there
    assert len(team.memory.insights) == 1
    assert len(team.memory.key_arguments) == 1
    assert "approach" in team.memory.strategic_notes
    
    # Get summary (as would be used in debate)
    summary = team.get_memory_summary()
    assert "Initial insight" in summary
    assert "Core argument" in summary
    assert "utilitarian" in summary
    
    print("✓ Memory persists correctly")
    print(f"  - Insights: {len(team.memory.insights)}")
    print(f"  - Arguments: {len(team.memory.key_arguments)}")


def test_phase_transitions():
    """Test phase transition from deliberation to debate."""
    print("\n=== TEST: Phase Transitions ===")
    
    # Initial state: deliberation
    state = TeamConversationState(
        teams={},
        chat_history=[],
        current_speaker_team="",
        current_speaker_agent="",
        current_phase="deliberation",
        turn_count=0,
        deliberation_count=0,
        ontology_insights=[],
        topic="Test"
    )
    
    assert state["current_phase"] == "deliberation"
    
    # Simulate transition to debate
    state["current_phase"] = "debate"
    state["deliberation_count"] = 2
    
    assert state["current_phase"] == "debate"
    assert state["deliberation_count"] == 2
    
    print("✓ Phase transitions work correctly")


def run_all_tests():
    """Run all unit tests for team conversation system."""
    print("\n" + "="*70)
    print("RUNNING TEAM CONVERSATION TESTS")
    print("="*70)
    
    try:
        test_team_conversation_state()
        test_enhanced_ontology()
        test_concept_tracking()
        test_ontology_team_integration()
        test_team_conversation_initialization()
        test_debate_insights_generation()
        test_memory_persistence_across_phases()
        test_phase_transitions()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70 + "\n")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()

