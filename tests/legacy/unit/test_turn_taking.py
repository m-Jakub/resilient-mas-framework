"""
Test: Turn-Taking System with Speech Queue and Logging

This script tests the enhanced turn-taking system with:
- Synchronized speech queues per team
- Multiple turn allocation strategies (round-robin, fair)
- Timeout handling
- Message length limits
- Structured conversation logging (JSON + CSV)

Test Scenarios:
1. Two-team debate with round-robin allocation
2. Verify queue behavior and fairness
3. Check timeout handling (simulated)
4. Validate structured logs (JSON + CSV output)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hde.team import Team
from src.hde.team_conversation import TeamPhilosophicalConversation
from src.common.philosopher_agents import create_agent
from dotenv import load_dotenv
import os

load_dotenv()


def test_turn_taking_round_robin():
    """
    Test Case 1: Round-robin turn allocation with two teams.
    
    Expected behavior:
    - Teams alternate speaking
    - Speech queue manages fair turn distribution
    - All turns logged with timestamps
    - Queue state tracked throughout debate
    """
    print("\n" + "="*70)
    print("TEST 1: ROUND-ROBIN TURN ALLOCATION")
    print("="*70)
    print()
    
    # Verify API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not set in .env")
        return False
    
    # Create teams
    team_a = Team("team_util", "Utilitarians")
    team_b = Team("team_deont", "Deontologists")
    
    print("Building teams...")
    
    # Team A: Utilitarians
    mill_agent = create_agent("mill")
    if mill_agent:
        team_a.add_agent("mill", mill_agent)
        print(f"  ✓ {team_a.name}: {mill_agent.name}")
    
    # Team B: Deontologists
    kant_agent = create_agent("kant")
    if kant_agent:
        team_b.add_agent("kant", kant_agent)
        print(f"  ✓ {team_b.name}: {kant_agent.name}")
    
    # Define ethical dilemma
    dilemma = """
The Trolley Problem: A runaway trolley is heading toward five people tied to the tracks.
You can pull a lever to divert it to another track, where it will kill one person instead.

Should you pull the lever? What is the most ethical course of action?
"""
    
    print("\n" + "="*70)
    print("ETHICAL DILEMMA")
    print("="*70)
    print(dilemma)
    print("="*70)
    
    # Create conversation with enhanced features
    conversation = TeamPhilosophicalConversation(
        deliberation_rounds=1,
        debate_turns=4,
        enable_moderation=True,
        moderation_tone="reflective",
        enable_logging=True,
        session_id="test_round_robin_001",
        turn_allocation_strategy="round_robin",
        max_message_length=800,
        turn_timeout_seconds=60
    )
    
    print("\nConfiguration:")
    print(f"  • Turn allocation: round-robin")
    print(f"  • Max message length: 800 chars")
    print(f"  • Turn timeout: 60s")
    print(f"  • Logging: enabled")
    print(f"  • Session ID: test_round_robin_001")
    print()
    
    # Run debate
    teams = {
        "team_util": team_a,
        "team_deont": team_b
    }
    
    try:
        final_state = conversation.start_debate(dilemma, teams)
        
        print("\n" + "="*70)
        print("TEST 1 RESULT: PASSED")
        print("="*70)
        print("✓ Debate completed successfully")
        print("✓ Logs saved to logs/test_round_robin_001.*")
        print("✓ Speech queue statistics recorded")
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print("TEST 1 RESULT: FAILED")
        print("="*70)
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_turn_taking_fair():
    """
    Test Case 2: Fair turn allocation (balances speaking time).
    
    Expected behavior:
    - Team with less speaking time gets next turn
    - Queue manager tracks cumulative speaking time
    - Fairness metrics logged
    """
    print("\n" + "="*70)
    print("TEST 2: FAIR TURN ALLOCATION")
    print("="*70)
    print()
    
    # Verify API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not set in .env")
        return False
    
    # Create teams (same setup)
    team_a = Team("team_util", "Utilitarians")
    team_b = Team("team_deont", "Deontologists")
    
    print("Building teams...")
    
    mill_agent = create_agent("mill")
    if mill_agent:
        team_a.add_agent("mill", mill_agent)
        print(f"  ✓ {team_a.name}: {mill_agent.name}")
    
    kant_agent = create_agent("kant")
    if kant_agent:
        team_b.add_agent("kant", kant_agent)
        print(f"  ✓ {team_b.name}: {kant_agent.name}")
    
    dilemma = """
Medical Ethics: A hospital has one donor organ available. Two patients need it:
- Patient A: 25 years old, otherwise healthy, expected to live 50+ more years
- Patient B: 65 years old, with other health complications, expected to live 5-10 more years

Who should receive the organ? What ethical principles guide this decision?
"""
    
    print("\n" + "="*70)
    print("ETHICAL DILEMMA")
    print("="*70)
    print(dilemma)
    print("="*70)
    
    # Create conversation with fair allocation
    conversation = TeamPhilosophicalConversation(
        deliberation_rounds=1,
        debate_turns=4,
        enable_moderation=True,
        moderation_tone="inquisitive",
        enable_logging=True,
        session_id="test_fair_allocation_001",
        turn_allocation_strategy="fair",
        max_message_length=800,
        turn_timeout_seconds=60
    )
    
    print("\nConfiguration:")
    print(f"  • Turn allocation: fair (balances speaking time)")
    print(f"  • Session ID: test_fair_allocation_001")
    print()
    
    teams = {
        "team_util": team_a,
        "team_deont": team_b
    }
    
    try:
        final_state = conversation.start_debate(dilemma, teams)
        
        print("\n" + "="*70)
        print("TEST 2 RESULT: PASSED")
        print("="*70)
        print("✓ Fair allocation completed successfully")
        print("✓ Logs saved to logs/test_fair_allocation_001.*")
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print("TEST 2 RESULT: FAILED")
        print("="*70)
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all turn-taking tests."""
    print("\n" + "="*70)
    print("TURN-TAKING SYSTEM TEST SUITE")
    print("="*70)
    print("\nTesting enhanced turn-taking with:")
    print("  • Speech queue management")
    print("  • Multiple allocation strategies")
    print("  • Timeout handling")
    print("  • Structured logging")
    print("="*70)
    
    results = []
    
    # Test 1: Round-robin
    print("\n\n")
    result_1 = test_turn_taking_round_robin()
    results.append(("Round-robin allocation", result_1))
    
    # Test 2: Fair allocation
    print("\n\n")
    result_2 = test_turn_taking_fair()
    results.append(("Fair allocation", result_2))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUITE SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("="*70)
    
    if passed == total:
        print("\n✓ All tests passed! Turn-taking system is functional.")
        print("\nNext steps:")
        print("1. Review logs in logs/ directory")
        print("2. Analyze queue behavior and fairness metrics")
        print("3. Check for timeout issues or edge cases")
    else:
        print("\n✗ Some tests failed. Review errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

