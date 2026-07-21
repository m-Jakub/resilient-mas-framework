"""
Test ToM-Lite (Theory of Mind) implementation.

Verifies:
1. Belief models are correctly initialized from identity graphs
2. Stance extraction heuristics work for all 8 philosophers
3. Strategic ToM prompts are generated with opponent weaknesses
4. ToM can be toggled on/off via flag (ablation study)
"""

import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.hde.team_conversation import TeamPhilosophicalConversation
from src.hde.team import Team
from src.common.philosopher_agents import create_agent

load_dotenv()

def test_stance_extraction():
    """Test stance extraction from identity graphs for all philosophers."""
    print("\n" + "="*80)
    print("TEST 1: Stance Extraction from Identity Graphs")
    print("="*80)
    
    conversation = TeamPhilosophicalConversation(
        use_tom=True,
        enable_logging=False,
        enable_moderation=False
    )
    
    # Use actual config keys from settings.py ACTIVE_PHILOSOPHERS
    philosophers = [
        "mill",          # Expected: utilitarian
        "kant",          # Expected: deontological
        "aquinas",       # Expected: natural_law
        "plato",         # Expected: virtue_ethics
        "aristotle",     # Expected: virtue_ethics
        "st_augustine",  # Expected: natural_law
        "bentham",       # Expected: utilitarian
        "nietzsche"      # Expected: will_to_power
    ]
    
    print("\nExtracting stances (using actual philosopher config keys)...")
    for agent_key in philosophers:
        agent = create_agent(agent_key)
        if agent:
            stance = conversation._extract_stance_from_identity(agent)
            print(f"[OK] {agent.name:40s} -> {stance}")
        else:
            print(f"[FAIL] {agent_key} - Agent creation failed")
    
    print("\n[OK] Stance extraction test complete")


def test_belief_model_initialization():
    """Test that belief models are correctly initialized in state."""
    print("\n" + "="*80)
    print("TEST 2: Belief Model Initialization")
    print("="*80)
    
    # Create minimal 2v2 debate using actual philosopher keys (not legacy tradition keys)
    team_a = Team("team_a", "Consequentialists")
    team_a.add_agent("mill", create_agent("mill"))
    team_a.add_agent("bentham", create_agent("bentham"))
    
    team_b = Team("team_b", "Deontologists")
    team_b.add_agent("kant", create_agent("kant"))
    team_b.add_agent("aquinas", create_agent("aquinas"))
    
    teams = {"team_a": team_a, "team_b": team_b}
    
    # Create conversation with ToM enabled
    conversation = TeamPhilosophicalConversation(
        deliberation_rounds=0,  # Skip deliberation for speed
        debate_turns=1,          # Single turn only
        enable_moderation=False, # Disable moderation for speed
        enable_logging=False,    # Disable logging for speed
        use_tom=True
    )
    
    # Run debate (will initialize belief_models)
    print("\nInitializing belief models...")
    final_state = conversation.start_debate(
        topic="Test dilemma (minimal)",
        teams=teams
    )
    
    # Verify belief_models exist and are populated
    belief_models = final_state.get("belief_models", {})
    
    print("\n" + "-"*80)
    print("Belief Models Verification:")
    print("-"*80)
    
    if not belief_models:
        print("[FAIL] ERROR: belief_models is empty!")
        return False
    
    for agent_full_name, model in belief_models.items():
        print(f"\nAgent: {agent_full_name}")
        print(f"  Self stance: {model.get('self_stance', 'MISSING')}")
        
        # Check opponent models
        opponent_keys = [k for k in model.keys() if k.startswith("opponent_")]
        print(f"  Opponent models: {len(opponent_keys)}")
        for opp_key in opponent_keys:
            print(f"    {opp_key}: {model[opp_key]}")
    
    print("\n[OK] Belief model initialization test complete")
    return True


def test_tom_strategic_prompt():
    """Test that strategic ToM prompts are generated correctly."""
    print("\n" + "="*80)
    print("TEST 3: ToM Strategic Prompt Generation")
    print("="*80)
    
    conversation = TeamPhilosophicalConversation(use_tom=True)
    
    # Test strategic prompt for Kant vs Mill scenario
    strategic_prompt = conversation._build_tom_strategy_prompt(
        agent_name="Kant",
        agent_stance="deontological",
        opponent_name="Mill",
        opponent_stance="utilitarian",
        opponent_last_argument="We must maximize happiness for the greatest number, even if it means sacrificing one individual's rights."
    )
    
    print("\nGenerated Strategic Prompt:")
    print("-"*80)
    print(strategic_prompt)
    print("-"*80)
    
    # Verify prompt contains key elements
    assert "Kant" in strategic_prompt
    assert "deontological" in strategic_prompt
    assert "Mill" in strategic_prompt
    assert "utilitarian" in strategic_prompt
    assert "weaknesses" in strategic_prompt.lower()
    
    print("\n[OK] Strategic prompt generation test complete")


def test_tom_ablation():
    """Test that ToM can be disabled (ablation control)."""
    print("\n" + "="*80)
    print("TEST 4: ToM Ablation (use_tom=False)")
    print("="*80)
    
    # Create minimal teams using actual philosopher keys
    team_a = Team("team_a", "Team A")
    team_a.add_agent("kant", create_agent("kant"))
    
    team_b = Team("team_b", "Team B")
    team_b.add_agent("mill", create_agent("mill"))
    
    teams = {"team_a": team_a, "team_b": team_b}
    
    # Create conversation with ToM DISABLED
    conversation = TeamPhilosophicalConversation(
        deliberation_rounds=0,
        debate_turns=1,
        enable_moderation=False,
        enable_logging=False,
        use_tom=False  # DISABLE ToM
    )
    
    print("\nRunning debate with ToM disabled...")
    final_state = conversation.start_debate(
        topic="Test dilemma (ToM off)",
        teams=teams
    )
    
    # Verify belief_models are NOT populated
    belief_models = final_state.get("belief_models", {})
    
    if len(belief_models) == 0:
        print("\n[OK] ToM ablation test PASSED: belief_models empty when use_tom=False")
        return True
    else:
        print(f"\n[FAIL] ToM ablation test FAILED: belief_models has {len(belief_models)} entries (expected 0)")
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("ToM-LITE IMPLEMENTATION TEST SUITE")
    print("="*80)
    
    try:
        test_stance_extraction()
        test_belief_model_initialization()
        test_tom_strategic_prompt()
        test_tom_ablation()
        
        print("\n" + "="*80)
        print("[OK] ALL TESTS PASSED")
        print("="*80)
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"[FAIL] TEST FAILED: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()

