"""
Quick demo: Generate ONLY baseline (Mill monologue) and display transcript.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from test_conceptual_coverage_conditions import run_baseline_condition

if __name__ == "__main__":
    # Trolley Problem dilemma
    dilemma = """
The Trolley Problem: A runaway trolley is headed towards five people tied to the tracks.
You are standing next to a lever. If you pull it, the trolley will be diverted onto a 
side track where one person is tied.

Do you pull the lever, actively causing one death to prevent five deaths?

Provide a thorough ethical analysis from your philosophical perspective.
""".strip()
    
    print("\n🎯 Generating BASELINE (Mill monologue) only...\n")
    
    # Generate baseline
    log_file = run_baseline_condition(dilemma, output_dir="logs/baselines")
    
    if log_file:
        print(f"\n✅ Saved to: {log_file}")
        print("\n📄 Want to see formatted transcript? Check the log file!")
