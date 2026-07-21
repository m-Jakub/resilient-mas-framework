"""
Quick script to generate transcript and SysAR analysis from existing JSON log.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from tests.run_single_sysar_test import save_readable_transcript
from src.hde.sysar_calculator import analyze_debate_sysar
import json

# Get path from command line argument
if len(sys.argv) < 2:
    print("Usage: python generate_sysar_from_log.py <path_to_json_log>")
    sys.exit(1)

json_log = Path(sys.argv[1])
if not json_log.exists():
    print(f"Error: File not found: {json_log}")
    sys.exit(1)

output_dir = Path("logs/experiments/sysar_experiments")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Processing: {json_log}")

# Generate transcript
print("\nGenerating transcript...")
transcript_file = save_readable_transcript(json_log, output_dir)

# Calculate SysAR
print("\nCalculating SysAR metrics...")
sysar_analysis = analyze_debate_sysar(
    debate_json_path=json_log,
    topic_type="trolley_problem"
)

# Print results (Dual-Metric System)
print(f"\n{'='*80}")
print("SYSAR RESULTS (DUAL-METRIC SYSTEM)")
print(f"{'='*80}")
print(f"\nPerturbation:")
print(f"  Type: {sysar_analysis['perturbation']['type']}")
print(f"  Injected at turn: {sysar_analysis['perturbation']['injected_at_turn']}")
print(f"  Text: {sysar_analysis['perturbation']['text'][:80]}...")

print(f"\n--- METRIC 1: SysAR (Strict Recovery to Original Topic) ---")
print(f"  Recovered: {sysar_analysis['sysar']['recovered']}")

if sysar_analysis['sysar']['recovered']:
    print(f"  Recovery turn: {sysar_analysis['sysar']['recovery_turn']}")
    print(f"  Recovery time: {sysar_analysis['sysar']['recovery_time']} turns")
    print(f"  Keywords found: {', '.join(sysar_analysis['sysar']['keywords_found'])}")
    print(f"  Message excerpt: {sysar_analysis['sysar']['recovery_message_excerpt'][:100]}...")
else:
    print("  ❌ System failed to return to original topic framing")

print(f"  SysAR Score: {sysar_analysis['sysar']['score']:.3f}")

print(f"\n--- METRIC 2: ArCo (Argumentative Coherence) ---")
print(f"  Coherent: {sysar_analysis['arco']['coherent']}")

if sysar_analysis['arco']['coherent']:
    print(f"  Coherence turn: {sysar_analysis['arco']['coherence_turn']}")
    print(f"  Coherence time: {sysar_analysis['arco']['coherence_time']} turns")
    print(f"  Keywords found: {', '.join(sysar_analysis['arco']['keywords_found'])}")
    print(f"  Message excerpt: {sysar_analysis['arco']['coherence_message_excerpt'][:100]}...")
else:
    print("  ❌ System completely derailed (catastrophic failure)")

print(f"  ArCo Score: {sysar_analysis['arco']['score']:.3f}")

print(f"\n--- OVERALL INTERPRETATION ---")
print(f"  {sysar_analysis['overall_interpretation']}")

# Save SysAR analysis - derive filename from input
experiment_name = json_log.stem  # e.g. "sysar_heterogeneous_clean_push_vs_lever_20251104_212950"
sysar_json = output_dir / f"{experiment_name}_analysis.json"
with open(sysar_json, 'w', encoding='utf-8') as f:
    json.dump(sysar_analysis, f, indent=2)

print(f"\n✅ SysAR analysis saved: {sysar_json}")
print(f"{'='*80}\n")
