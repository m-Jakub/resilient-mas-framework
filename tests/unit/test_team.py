"""
Unit tests for the Team module.
Tests basic team operations, memory management, and agent coordination.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hde.team import Team, TeamMemory
from src.common.philosopher_agents import create_agent


class MockAgent:
    """Mock agent for testing team operations without full RAG setup."""
    
    def __init__(self, key: str, name: str):
        self.key = key
        self.name = name
        self.school = key
        self.persona = f"Mock {name}"
    
    def respond(self, topic: str, chat_history: list, ontology_insights: list = None):
        """Mock response method."""
        return f"This is {self.name}'s response to: {topic[:50]}..."


def test_team_creation():
    """Test basic team creation."""
    print("\n=== TEST: Team Creation ===")
    team = Team("team_a", "Team Alpha")
    
    assert team.team_id == "team_a"
    assert team.name == "Team Alpha"
    assert len(team.agents) == 0
    assert team.leader is None
    print("✓ Team created successfully")
    return team


def test_add_agents():
    """Test adding agents to a team."""
    print("\n=== TEST: Adding Agents ===")
    team = Team("team_b", "Team Beta")
    
    # Create mock agents
    agent1 = MockAgent("util_1", "Utilitarian Agent")
    agent2 = MockAgent("deont_1", "Deontological Agent")
    
    team.add_agent("util_1", agent1)
    team.add_agent("deont_1", agent2)
    
    assert len(team.agents) == 2
    assert "util_1" in team.agents
    assert "deont_1" in team.agents
    print("✓ Agents added successfully")
    return team


def test_remove_agent():
    """Test removing agents from a team."""
    print("\n=== TEST: Removing Agents ===")
    team = Team("team_c", "Team Gamma")
    
    agent1 = MockAgent("agent_1", "Agent 1")
    agent2 = MockAgent("agent_2", "Agent 2")
    
    team.add_agent("agent_1", agent1)
    team.add_agent("agent_2", agent2)
    
    # Remove one agent
    result = team.remove_agent("agent_1")
    assert result == True
    assert len(team.agents) == 1
    assert "agent_1" not in team.agents
    
    # Try to remove non-existent agent
    result = team.remove_agent("nonexistent")
    assert result == False
    print("✓ Agent removal works correctly")


def test_team_memory():
    """Test team memory operations."""
    print("\n=== TEST: Team Memory ===")
    memory = TeamMemory("team_test")
    
    # Test adding insights
    memory.add_insight("First insight about utility", "Agent A")
    memory.add_insight("Second insight about duty", "Agent B")
    assert len(memory.insights) == 2
    
    # Test adding arguments
    memory.add_argument("Argument for consequentialism", "Agent A")
    assert len(memory.key_arguments) == 1
    
    # Test adding concepts
    memory.add_ethical_concept("utility")
    memory.add_ethical_concept("duty")
    memory.add_ethical_concept("utility")  # Duplicate
    assert len(memory.ethical_concepts) == 2  # No duplicates
    
    # Test strategic notes
    memory.set_strategic_note("approach", "focus on long-term consequences")
    assert "approach" in memory.strategic_notes
    
    # Test summary generation
    summary = memory.get_summary()
    assert "Team team_test Shared Memory" in summary
    assert "utility" in summary
    assert "duty" in summary
    
    print("✓ Team memory operations work correctly")
    print(f"\nMemory summary:\n{summary}")


def test_set_leader():
    """Test setting team leader."""
    print("\n=== TEST: Setting Team Leader ===")
    team = Team("team_d", "Team Delta")
    
    agent1 = MockAgent("leader_1", "Leader Agent")
    team.add_agent("leader_1", agent1)
    
    team.set_leader("leader_1")
    assert team.leader == "leader_1"
    print("✓ Team leader set successfully")


def test_memory_summary():
    """Test team memory summary generation."""
    print("\n=== TEST: Memory Summary ===")
    team = Team("team_e", "Team Epsilon")
    
    # Add some data to memory
    team.memory.add_insight("Insight 1", "Agent A")
    team.memory.add_argument("Argument 1", "Agent B")
    team.memory.add_ethical_concept("virtue")
    team.memory.set_strategic_note("focus", "human dignity")
    
    summary = team.get_memory_summary()
    
    assert "Team Epsilon" in summary or "team_e" in summary
    assert "Insight 1" in summary
    assert "Argument 1" in summary
    assert "virtue" in summary
    assert "focus" in summary
    
    print("✓ Memory summary generated correctly")
    print(f"\n{summary}")


def test_memory_clear():
    """Test clearing team memory."""
    print("\n=== TEST: Clearing Memory ===")
    memory = TeamMemory("team_clear")
    
    memory.add_insight("Test insight", "Agent")
    memory.add_argument("Test argument", "Agent")
    memory.add_ethical_concept("test_concept")
    
    assert len(memory.insights) > 0
    
    memory.clear()
    
    assert len(memory.insights) == 0
    assert len(memory.key_arguments) == 0
    assert len(memory.ethical_concepts) == 0
    assert len(memory.strategic_notes) == 0
    
    print("✓ Memory cleared successfully")


def test_update_memory_from_deliberation():
    """Test updating memory during deliberation."""
    print("\n=== TEST: Update Memory from Deliberation ===")
    team = Team("team_f", "Team Zeta")
    
    agent1 = MockAgent("agent_1", "Agent Alpha")
    team.add_agent("agent_1", agent1)
    
    # Update with different content types
    team.update_memory_from_deliberation("agent_1", "New insight", "insight")
    team.update_memory_from_deliberation("agent_1", "New argument", "argument")
    team.update_memory_from_deliberation("agent_1", "utility", "concept")
    
    assert len(team.memory.insights) == 1
    assert len(team.memory.key_arguments) == 1
    assert "utility" in team.memory.ethical_concepts
    
    print("✓ Memory updated correctly from deliberation")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "="*60)
    print("RUNNING TEAM MODULE UNIT TESTS")
    print("="*60)
    
    try:
        test_team_creation()
        test_add_agents()
        test_remove_agent()
        test_team_memory()
        test_set_leader()
        test_memory_summary()
        test_memory_clear()
        test_update_memory_from_deliberation()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60 + "\n")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
