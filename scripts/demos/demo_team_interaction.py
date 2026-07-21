"""
Demo: Minimal Team Interaction
Demonstrates agents within a team conducting internal deliberation.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hde.team import Team
from src.common.philosopher_agents import create_agent

load_dotenv()


def demo_team_deliberation():
    """
    Demonstrates a team of agents conducting internal deliberation (narada)
    before entering a debate.
    """
    print("\n" + "="*70)
    print("DEMO: Team Internal Deliberation (Narada)")
    print("="*70 + "\n")
    
    # Create a team
    team = Team("team_consequentialist", "Consequentialist Team")
    
    print("Creating team with real philosopher agents...")
    print("-" * 70)
    
    # Add real agents to the team
    mill_agent = create_agent("mill")
    if mill_agent:
        team.add_agent("mill", mill_agent)
    
    print("\n" + "-" * 70)
    print(f"Team '{team.name}' created with {len(team.agents)} agent(s)")
    print("-" * 70)
    
    # Define an ethical dilemma
    dilemma = """
The Trolley Problem: A runaway trolley is speeding toward five people tied to the tracks.
You can pull a lever to divert it to another track, where it will kill one person instead.
What should you do?
"""
    
    print(f"\nDilemma:\n{dilemma}")
    
    # Conduct internal team deliberation
    print("\n" + "="*70)
    print("PHASE 1: Internal Team Deliberation (Narada)")
    print("="*70)
    
    team.internal_deliberation(topic=dilemma, rounds=1)
    
    # Display team memory after deliberation
    print("\n" + "="*70)
    print("TEAM MEMORY AFTER DELIBERATION")
    print("="*70)
    print(team.get_memory_summary())
    
    # Show what the team learned
    print("\n" + "="*70)
    print("KEY OUTCOMES")
    print("="*70)
    print(f"✓ Insights collected: {len(team.memory.insights)}")
    print(f"✓ Arguments formulated: {len(team.memory.key_arguments)}")
    print(f"✓ Ethical concepts identified: {len(team.memory.ethical_concepts)}")
    print(f"✓ Deliberation entries: {len(team.memory.deliberation_history)}")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70 + "\n")
    
    return team


def demo_two_teams():
    """
    Demonstrates two teams preparing for debate through internal deliberation.
    """
    print("\n" + "="*70)
    print("DEMO: Two Teams Preparing for Debate")
    print("="*70 + "\n")
    
    # Team 1: Consequentialists
    team1 = Team("team_consequentialist", "Consequentialist Team")
    mill_agent = create_agent("mill")
    if mill_agent:
        team1.add_agent("mill", mill_agent)
    
    # Team 2: Deontologists
    team2 = Team("team_deontologist", "Deontological Team")
    kant_agent = create_agent("kant")
    if kant_agent:
        team2.add_agent("kant", kant_agent)
    
    dilemma = """
Should a doctor lie to a terminally ill patient about their prognosis to maintain their hope and quality of life?
"""
    
    print(f"\nDilemma:\n{dilemma}")
    
    # Both teams deliberate internally
    print("\n" + "="*70)
    print("TEAM 1: Internal Deliberation")
    print("="*70)
    team1.internal_deliberation(topic=dilemma, rounds=1)
    
    print("\n" + "="*70)
    print("TEAM 2: Internal Deliberation")
    print("="*70)
    team2.internal_deliberation(topic=dilemma, rounds=1)
    
    # Compare team memories
    print("\n" + "="*70)
    print("COMPARISON: Team Memories")
    print("="*70 + "\n")
    
    print("--- TEAM 1 MEMORY ---")
    print(team1.get_memory_summary())
    
    print("\n--- TEAM 2 MEMORY ---")
    print(team2.get_memory_summary())
    
    print("\n" + "="*70)
    print("NEXT PHASE: Teams would now enter inter-team debate")
    print("(To be implemented in Block 2)")
    print("="*70 + "\n")
    
    return team1, team2


if __name__ == "__main__":
    import os
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please set your Google API key in the .env file.")
        exit(1)
    
    print("\nChoose demo:")
    print("1. Single team deliberation")
    print("2. Two teams preparing for debate")
    
    choice = input("\nEnter choice (1 or 2, default=1): ").strip()
    
    if choice == "2":
        demo_two_teams()
    else:
        demo_team_deliberation()
