import os
import argparse
import sys
from pathlib import Path

# Ensure src/ is on sys.path for direct execution
repo_root = Path(__file__).resolve().parent
src_path = repo_root / "src"
if src_path.exists():
    src_str = str(src_path)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
from dotenv import load_dotenv

# --- Modular architecture imports ---
from src.hde.legacy_conversation import PhilosophicalConversation  # Article 1 (HDE)
from src.hde.baseline_tutor import run_baseline_tutor            # Article 1 (HDE)
from src.kg_cfr.aegis_orchestrator import run_tripartite_standoff # Article 2 (KG-CFR)
# ------------------------------------

import json
import yaml
from datetime import datetime
from config.settings import DILEMMAS_AEGIS_FILE, DILEMMAS_FILE

def load_dilemmas(path: str | Path = DILEMMAS_FILE):
    """Load dilemmas from a YAML file."""
    try:
        with open(Path(path), "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("dilemmas", [])
    except FileNotFoundError:
        print("Dilemmas file not found.")
        return []
    except Exception as e:
        print(f"Error loading dilemmas: {e}")
        return []

def interactive_setup():
    """Interactive setup for the conversation."""
    print("Welcome to the Ethical Debate Simulator!")
    print("=" * 60)
    
    # --- Shared infrastructure import ---
    from src.common.philosopher_agents import list_available_philosophers
    # ------------------------------------
    
    print("\nAvailable ethical agents:")
    philosophers = list_available_philosophers()
    for key, description in philosophers.items():
        print(f"{key}")
    
    conversation = PhilosophicalConversation()

    # Let user choose number of agents
    num_input = input("\nHow many agents for the conversation? (default: 2): ").strip()
    num_agents = int(num_input) if num_input.isdigit() and int(num_input) > 0 else 2
    print(f"\nSelect {num_agents} agents for the conversation:")
    participants_keys = list(philosophers.keys())
    selected_agents = []
    for i in range(num_agents):
        while True:
            choice = input(f"Enter agent {i+1} (from list above): ").strip().lower()
            if choice in participants_keys and choice not in selected_agents:
                conversation.add_participant(choice)
                selected_agents.append(choice)
                break
            elif choice in selected_agents:
                print("Agent already selected. Please choose a different one.")
            else:
                print("Invalid choice. Please select from the list.")
    
    # Let user choose how to set the topic
    print("\nHow would you like to set the topic?")
    print("1. Choose from a list of predefined dilemmas")
    print("2. Enter my own custom topic")
    topic_choice = input("Enter choice (1 or 2): ").strip()
    topic = ""
    if topic_choice == '1':
        dilemmas = load_dilemmas()
        if not dilemmas:
            return
        print("\nChoose an ethical dilemma:")
        for i, d in enumerate(dilemmas):
            print(f"{i+1}. {d['title']}")
        try:
            idx = int(input(f"Enter choice (1-{len(dilemmas)}): ")) - 1
            if not 0 <= idx < len(dilemmas):
                raise ValueError
            topic = dilemmas[idx]['prompt']
        except (ValueError, IndexError):
            print("Invalid choice. Exiting.")
            return
    elif topic_choice == '2':
        topic = input("\nPlease enter the topic for the debate: ").strip()
        if not topic:
            print("Topic cannot be empty. Exiting.")
            return
    else:
        print("Invalid choice. Exiting.")
        return
    
    turns_input = input(f"\nHow many total turns for the debate? (default: 6): ").strip()
    total_turns = int(turns_input) if turns_input.isdigit() and int(turns_input) > 0 else 6

    return conversation, topic, total_turns

def quick_demo():
    """Quick demo with a predefined setup."""
    print("Running demo: Utilitarian vs. Deontological vs. Christian Ethics on 'The Trolley Problem'")
    
    # The new class takes total_turns in its constructor
    conversation = PhilosophicalConversation(total_turns=6)

    participants = ["mill", "kant", "aquinas"]
    
    topic = ("The trolley problem: A runaway trolley is headed towards five people. "
             "You can pull a lever to divert it, killing one person instead. "
             "What is the ethical course of action?")
    
    # The new start_conversation takes the topic and a list of participant keys
    return conversation.start_conversation(topic, participants)

def main():
    load_dotenv()
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Philosopher Agents: A/B Testing Platform for Ethical Education"
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['baseline', 'aegis', 'legacy'],
        default='aegis',
        help="Interaction mode: 'baseline' (single tutor), 'aegis' (Tripartite Standoff), or 'legacy' (old system)"
    )
    parser.add_argument(
        '--dilemma',
        type=str,
        help="Directly specify ethical dilemma (skips interactive selection)"
    )
    parser.add_argument(
        '--max-turns',
        type=int,
        default=10,
        help="Maximum turns for baseline mode or debate turns for debate mode"
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Optional path to save the interaction log as JSON."
    )
    parser.add_argument(
        '--use-tom',
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=True,
        help="Enable Theory of Mind (ToM-Lite) for strategic opponent modeling (default: True)"
    )
    parser.add_argument(
        '--cfr-mode',
        type=str,
        choices=[
            'no_cfr_baseline', 'cfr_no_kg', 'kg_cfr_full',   # canonical
            'none', 'prompt-only', 'knowledge-grounded',      # legacy aliases
        ],
        default='no_cfr_baseline',
        help=(
            "CFR mode (default: no_cfr_baseline). "
            "Canonical: no_cfr_baseline | cfr_no_kg | kg_cfr_full. "
            "Legacy aliases also accepted: none | prompt-only | knowledge-grounded."
        )
    )
    parser.add_argument(
        '--use-idrag',
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=True,
        help="Enable ID-RAG identity context (default: True)"
    )
    parser.add_argument(
        '--use-adaptive-idrag',
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        default=True,
        help="Enable adaptive ID-RAG (default: True)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("AEGIS POLICY MODULES - TRIPARTITE STANDOFF")
    print("="*80)
    print(f"Mode: {args.mode.upper()}")
    print("="*80 + "\n")
    
    # Legacy mode (original interactive system)
    if args.mode == 'legacy':
        print("\n" + "="*80)
        print("⚠️  WARNING: LEGACY MODE - DEPRECATED")
        print("="*80)
        print("You are using the old single-agent conversation system.")
        print("For new features (AEGIS Tripartite Standoff + KG-CFR),")
        print("use '--mode=aegis' instead.")
        print("="*80 + "\n")
        
        print("Running legacy interactive mode...")
        print("Choose mode:")
        print("1. Interactive setup")
        print("2. Quick demo")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            conversation, topic, total_turns = interactive_setup()
            conversation.start_conversation(topic, total_turns)
        elif choice == "2":
            quick_demo()
        else:
            print("Invalid choice. Starting interactive setup...")
            conversation, topic, total_turns = interactive_setup()
            conversation.start_conversation(topic, total_turns)
        return
    
    # Get dilemma
    if args.dilemma:
        dilemma = args.dilemma
    else:
        # Interactive dilemma selection
        dilemmas_path = DILEMMAS_AEGIS_FILE if args.mode == "aegis" else DILEMMAS_FILE
        dilemmas = load_dilemmas(dilemmas_path)
        if not dilemmas:
            print("No dilemmas found. Exiting.")
            return
        
        print("Available ethical dilemmas:")
        for i, d in enumerate(dilemmas, 1):
            print(f"{i}. {d['title']}")
        
        try:
            choice = int(input(f"\nSelect dilemma (1-{len(dilemmas)}): "))
            dilemma = dilemmas[choice - 1]['prompt']
        except (ValueError, IndexError):
            print("Invalid choice. Using default (AEGIS blackout scenario).")
            dilemma = dilemmas[0]['prompt'] if dilemmas else (
                "A massive metropolis has lost the power grid; only 30% reserves remain. "
                "The AEGIS committee must allocate power under severe uncertainty."
            )
    
    # Run interaction (Baseline OR AEGIS)
    if args.mode == 'baseline':
        interaction_log = run_baseline_tutor(dilemma, max_turns=args.max_turns)
    elif args.mode == 'aegis':
        standoff_state = run_tripartite_standoff(
            crisis_context=dilemma,
            debate_turns=args.max_turns,
            cfr_mode=args.cfr_mode,
            shock_type="auto",
            shock_seed=None,
            use_id_rag=args.use_idrag,
            use_adaptive_idrag=args.use_adaptive_idrag,
            session_id=None,
        )
        
        interaction_log = {
            "crisis_context": standoff_state.get("crisis_context", ""),
            "turn_count": standoff_state.get("turn_count", 0),
            "cfr_mode": standoff_state.get("cfr_mode", ""),
            "chat_history": [
                {
                    "speaker": getattr(m, "name", getattr(m, "type", "unknown")), 
                    "content": getattr(m, "content", str(m))
                }
                for m in standoff_state.get("chat_history", [])
            ],
            "phase4_synthesis": {
                "synthesis_final": standoff_state.get("phase4_synthesis", {}).get("synthesis_final", ""),
                "self_correction_count": standoff_state.get("phase4_synthesis", {}).get("self_correction_count", 0),
                "provenance_fidelity_scores": standoff_state.get("phase4_synthesis", {}).get("provenance_fidelity_scores", [])
            }
        }
    
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if args.mode == 'baseline':
            output_dir = Path("logs/baselines")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"baseline_{timestamp}.json"
        elif args.mode == 'aegis':
            output_dir = Path("logs/experiments")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"aegis_{timestamp}.json"
        else:
            output_dir = Path("logs/production")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"legacy_{timestamp}.json"
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(interaction_log, f, indent=2, ensure_ascii=False)
        print(f"\nInteraction log saved to: {output_path}")
    except Exception as e:
        print(f"Failed to save interaction log to {output_path}: {e}")

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in environment variables.")
        print("Please set your Google API key in the .env file.")
        exit(1)
    
    main()