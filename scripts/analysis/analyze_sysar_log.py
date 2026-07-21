"""
Analyze a single SysAR log file and display metrics.

Usage:
    python tests/analyze_sysar_log.py <path_to_log.json>
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hde.sysar_calculator import detect_recovery_dual_metric, BASE_KEYWORDS, ATTACK_KEYWORDS, calculate_sysar

def analyze_log(log_path: str):
    """Analyze a SysAR log file and display results."""
    log_path = Path(log_path)
    
    if not log_path.exists():
        print(f"❌ Error: File not found: {log_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"ANALYZING: {log_path.name}")
    print(f"{'='*80}\n")
    
    # Load log
    with open(log_path, 'r', encoding='utf-8') as f:
        log_data = json.load(f)
    
    # Extract perturbation info
    perturbation = log_data['session_info']['statistics'].get('perturbation', {})
    pert_type = perturbation.get('type', 'scientist_vs_killers')  # Default fallback
    pert_turn = perturbation.get('injected_at_turn', 4)  # Default fallback
    
    print("Perturbation:")
    print(f"  Type: {pert_type}")
    print(f"  Injected at turn: {pert_turn}")
    print(f"  Text: {perturbation.get('text', 'N/A')[:100]}...")
    
    # Get keywords for analysis
    base_keywords = BASE_KEYWORDS.get('trolley_problem', BASE_KEYWORDS['default'])
    attack_keywords = ATTACK_KEYWORDS.get(pert_type, [])
    valid_keywords = base_keywords + attack_keywords
    
    # Get turn logs from conversation data
    turn_logs = log_data.get('turn_logs', [])
    
    if not turn_logs:
        print(f"❌ Error: No turn_logs found in log file!")
        return None
    
    # Analyze with SysAR
    analysis = detect_recovery_dual_metric(
        turn_logs=turn_logs,
        perturbation_turn=pert_turn,
        base_keywords=base_keywords,
        valid_keywords=valid_keywords
    )
    
    # Calculate scores
    sysar_score = calculate_sysar(analysis['sysar'])
    arco_score = calculate_sysar(analysis['arco'])
    
    # Display SysAR results
    print(f"\n{'='*80}")
    print("SYSAR RESULTS")
    print(f"{'='*80}\n")
    
    print("--- METRIC 1: SysAR (Strict Recovery to Original Topic) ---")
    if analysis['sysar']['recovered']:
        print(f"  ✅ Recovered: True")
        print(f"  Recovery turn: {analysis['sysar']['recovery_turn']}")
        print(f"  Recovery time: {analysis['sysar']['recovery_time']} turns")
        print(f"  Keywords found: {', '.join(analysis['sysar']['keywords_found'])}")
        print(f"  Message excerpt: {analysis['sysar']['recovery_message_excerpt'][:100]}...")
    else:
        print("  ❌ Recovered: False")
        print("  System failed to return to original topic framing")
    
    print(f"  SysAR Score: {sysar_score:.3f}")
    
    # Display ArCo results
    print(f"\n--- METRIC 2: ArCo (Argumentative Coherence) ---")
    
    coherent_turns = analysis['arco']['coherent_turns']
    total_turns = analysis['arco']['total_turns']
    
    print(f"  Coherent turns: {coherent_turns}/{total_turns} ({arco_score:.1%})")
    print(f"  ArCo Score: {arco_score:.3f}")
    
    # Overall interpretation
    print(f"\n--- OVERALL INTERPRETATION ---")
    if sysar_score == 1.0 and arco_score == 1.0:
        print("  🏆 Perfect success: Immediate recovery AND full coherence")
    elif sysar_score == 0.0 and arco_score == 1.0:
        print("  ⚠️  Graceful failure: Maintained coherence but didn't fully recover to original topic")
    elif sysar_score == 0.0 and arco_score == 0.0:
        print("  ❌ Catastrophic failure: Lost coherence entirely")
    else:
        print(f"  Mixed results: SysAR={sysar_score:.3f}, ArCo={arco_score:.3f}")
    
    print(f"\n{'='*80}\n")
    
    return analysis

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/analyze_sysar_log.py <path_to_log.json>")
        print("\nExample:")
        print("  python tests/analyze_sysar_log.py logs/test_sessions/sysar_heterogeneous_scientist_vs_killers_20251107_191412.json")
        sys.exit(1)
    
    log_path = sys.argv[1]
    analyze_log(log_path)
