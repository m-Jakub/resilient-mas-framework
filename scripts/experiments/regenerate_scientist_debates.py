"""
Regenerate scientist_vs_killers debates with Topic Anchor for human study.

Generates TWO golden transcripts:
- Group B (Homo): 4 Virtue Ethics philosophers
- Group C (Hetero): 2 Utilitarian vs 2 Virtue Ethics

These will be used in Google Forms experiment comparing:
- Group A: Baseline (vanilla LLM)
- Group B: Homogeneous debate
- Group C: Heterogeneous debate
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.run_single_sysar_test import run_single_sysar_test

if __name__ == "__main__":
    import sys
    
    print("=" * 80)
    print("REGENERATING SCIENTIST_VS_KILLERS DEBATES WITH TOPIC ANCHOR")
    print("=" * 80)
    print("\nThese transcripts will be used for human study (Google Forms experiment)")
    print("Testing whether Topic Anchor improves debate coherence post-perturbation\n")
    
    # Check command line argument to run one at a time
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python tests/regenerate_scientist_debates.py hetero")
        print("  python tests/regenerate_scientist_debates.py homo")
        sys.exit(1)
    
    config = sys.argv[1].lower()
    
    if config == "hetero":
        # Heterogeneous (Group C for human study)
        print("\n" + "=" * 80)
        print("HETEROGENEOUS DEBATE (Utilitarian vs Virtue Ethics)")
        print("=" * 80)
        
        run_single_sysar_test(
            experiment_type="heterogeneous_clean",
            perturbation_type="scientist_vs_killers"
        )
        
        print("\n✅ Heterogeneous debate complete!")
        print("Next: Run `python tests/regenerate_scientist_debates.py homo`")
        
    elif config == "homo":
        # Homogeneous (Group B for human study)
        print("\n" + "=" * 80)
        print("HOMOGENEOUS DEBATE (4 Virtue Ethics philosophers)")
        print("=" * 80)
        
        run_single_sysar_test(
            experiment_type="homogeneous",
            perturbation_type="scientist_vs_killers"
        )
        
        print("\n✅ Homogeneous debate complete!")
        print("Both debates regenerated with Topic Anchor!")
        
    else:
        print(f"❌ Unknown config: {config}")
        print("Use 'hetero' or 'homo'")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("Next steps:")
    print("1. Check logs/test_sessions/ for new transcripts with today's timestamp")
    print("2. Run SysAR analysis on both logs")
    print("3. Export transcripts to human-readable format for Google Forms")
    print("4. Compare ArCo scores: Hetero vs Homo vs Baseline")
    print("\nExpected improvement: Topic Anchor should reduce post-perturbation drift")
    print("=" * 80)
