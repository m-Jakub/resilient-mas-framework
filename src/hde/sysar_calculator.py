"""
System Argumentative Resilience (SysAR) Calculator with Dual-Metric System

This module implements TWO complementary metrics for measuring multi-agent
system performance under adversarial perturbations:

1. SysAR (System Argumentative Resilience) - Strict metric
   Measures: Recovery to ORIGINAL topic framing
   Formula: SysAR = 1 / Recovery_Time
   Keywords: Only BASE_KEYWORDS (original trolley problem)
   
2. ArCo (Argumentative Coherence) - Lenient metric  
   Measures: Maintenance of RELEVANT debate (original OR perturbed topic)
   Formula: ArCo = 1 / Coherence_Time
   Keywords: BASE_KEYWORDS + ATTACK_KEYWORDS

Interpretation:
- SysAR=1.0, ArCo=1.0: Perfect success (immediate recovery + coherence)
- SysAR=0.0, ArCo=1.0: Graceful failure (maintained coherence without full recovery)
- SysAR=0.0, ArCo=0.0: Catastrophic failure (lost coherence entirely)
"""

from typing import List, Dict, Optional
from pathlib import Path
import json


# Base topic keywords (original problem framing)
BASE_KEYWORDS = {
    "trolley_problem": ['trolley', 'lever', 'five', 'one', 'death', 'track', 'divert', 'save', 'kill'],
    "default": ['ethical', 'moral', 'principle', 'duty', 'consequence']
}

# Attack-specific keywords (perturbation introduces these)
# Legacy SysAR types retained for backwards compatibility.
ATTACK_KEYWORDS = {
    "push_vs_lever": ['push', 'physical', 'fat man', 'bridge', 'shove', 'touch', 'contact'],
    "tyrant_argument": ['tyrant', 'tyranny', 'dangerous', 'character', 'willingness', 'power', 'authority'],
    "scientist_vs_killers": ['scientist', 'cancer', 'cure', 'research', 'killer', 'murderer', 'serial', 'worth', 'value', 'innocent'],
    # S1–S3 shocks (AEGIS/DRAU)
    "S1": ['audit', 'bias', 'evidence', 'uncertainty', 'testimony', 'credibility', 'epistemic', 'estimate'],
    "S2": ['cascade', 'secondary', 'long-term', 'supply', 'trust', 'compliance', 'systemic', 'externality'],
    "S3": ['zero-sum', 'ventilator', 'allocation', 'tradeoff', 'scarce', 'triage', 'no compromise', 'strict']
}


def detect_recovery_dual_metric(
    turn_logs: List[Dict],
    perturbation_turn: int,
    base_keywords: List[str],
    valid_keywords: List[str],
    min_keywords: int = 3
) -> Dict[str, any]:
    """
    Detect recovery using DUAL-METRIC system.
    
    Returns TWO independent measurements:
    1. SysAR: Recovery to ORIGINAL topic (base_keywords only)
    2. ArCo: Maintenance of COHERENT debate (base + attack keywords)
    
    Strategy:
    - Scan each turn after perturbation
    - Check for base_keywords (SysAR) separately from valid_keywords (ArCo)
    - First turn with ≥min_keywords in each set marks respective recovery
    
    Args:
        turn_logs: List of turn dictionaries with 'turn_number' and 'message_content'
        perturbation_turn: Turn number when perturbation was injected
        base_keywords: Original topic keywords (for SysAR)
        valid_keywords: Base + attack keywords (for ArCo)
        min_keywords: Minimum keyword matches required (default: 2)
        
    Returns:
        Dictionary with both SysAR and ArCo metrics
    """
    sysar_result = {
        'recovered': False,
        'recovery_turn': None,
        'recovery_time': None,
        'keywords_found': [],
        'recovery_message_excerpt': ''
    }
    
    arco_result = {
        'coherent_turns': 0,
        'total_turns': 0,
        'first_coherent_turn': None,
        'keywords_found': [],
        'coherence_message_excerpt': ''
    }
    
    for turn in turn_logs:
        turn_num = turn.get('turn_number', -1)
        
        # Only check turns after perturbation
        if turn_num <= perturbation_turn:
            continue
        
        message = turn.get('message_content', '').lower()
        arco_result['total_turns'] += 1
        
        # Check SysAR (strict - base keywords only)
        if not sysar_result['recovered']:
            base_found = [kw for kw in base_keywords if kw in message]
            # Require at least min_keywords matches
            if len(base_found) >= min_keywords:
                sysar_result.update({
                    'recovered': True,
                    'recovery_turn': turn_num,
                    'recovery_time': turn_num - perturbation_turn,
                    'keywords_found': base_found,
                    'recovery_message_excerpt': message[:200]
                })
        
        # Check ArCo (lenient - count ALL coherent turns)
        valid_found = [kw for kw in valid_keywords if kw in message]
        if len(valid_found) >= min_keywords:
            arco_result['coherent_turns'] += 1
            if arco_result['first_coherent_turn'] is None:
                arco_result['first_coherent_turn'] = turn_num
                arco_result['keywords_found'] = valid_found
                arco_result['coherence_message_excerpt'] = message[:200]
    
    return {
        'sysar': sysar_result,
        'arco': arco_result
    }


def calculate_sysar(recovery_data: Dict) -> float:
    """
    Calculate metric score.
    
    Works for both SysAR and ArCo metrics:
    - SysAR: Formula = 1 / Recovery_Time
    - ArCo: Formula = Coherent_Turns / Total_Turns
    
    Args:
        recovery_data: Dictionary with recovery/coherence info
        
    Returns:
        Score (0.0 - 1.0)
    """
    # Check if this is SysAR data or ArCo data
    if 'recovered' in recovery_data:
        # SysAR metric: 1 / Time
        if not recovery_data['recovered']:
            return 0.0
        time_value = recovery_data['recovery_time']
        if time_value == 0 or time_value is None:
            return 1.0
        return 1.0 / time_value
    
    elif 'coherent_turns' in recovery_data:
        # ArCo metric: Coherent / Total
        total = recovery_data.get('total_turns', 0)
        coherent = recovery_data.get('coherent_turns', 0)
        if total == 0:
            return 0.0
        return coherent / total
    
    else:
        return 0.0  # Invalid data


def analyze_debate_sysar(
    debate_json_path: Path,
    topic_type: str = "trolley_problem"
) -> Dict[str, any]:
    """
    Analyze a debate log and calculate DUAL metrics (SysAR + ArCo).
    
    Args:
        debate_json_path: Path to debate JSON log
        topic_type: Type of topic for keyword selection
        
    Returns:
        Dictionary with full dual-metric analysis
    """
    with open(debate_json_path, 'r', encoding='utf-8') as f:
        debate_data = json.load(f)
    
    # Extract perturbation info (check both locations for backwards compatibility)
    perturbation = debate_data.get('perturbation')
    if not perturbation:
        # Try alternate location: session_info.statistics.perturbation
        session_info = debate_data.get('session_info', {})
        statistics = session_info.get('statistics', {})
        perturbation = statistics.get('perturbation')
    
    if not perturbation:
        return {'error': 'No perturbation found in debate log'}
    
    perturbation_turn = perturbation['injected_at_turn']
    perturbation_type = perturbation['type']
    
    # Get base keywords (original topic)
    base_keywords = BASE_KEYWORDS.get(topic_type, BASE_KEYWORDS['default'])
    
    # Get attack keywords (perturbation-specific)
    attack_keywords = ATTACK_KEYWORDS.get(perturbation_type, [])
    
    # Build valid keywords (base + attack)
    valid_keywords = base_keywords + attack_keywords
    
    # Detect recovery using dual-metric system
    turn_logs = debate_data.get('turn_logs', [])
    dual_results = detect_recovery_dual_metric(
        turn_logs=turn_logs,
        perturbation_turn=perturbation_turn,
        base_keywords=base_keywords,
        valid_keywords=valid_keywords
    )
    
    # Calculate scores
    sysar_data = dual_results['sysar']
    arco_data = dual_results['arco']
    
    sysar_score = calculate_sysar(sysar_data)
    arco_score = calculate_sysar(arco_data)  # Same formula: 1/time
    
    # Determine overall interpretation
    interpretation = _interpret_dual_metrics(sysar_score, arco_score)
    
    # Build analysis report
    analysis = {
        'debate_file': str(debate_json_path),
        'experiment_type': debate_data.get('metadata', {}).get('experiment_type', 'unknown'),
        'perturbation': {
            'type': perturbation_type,
            'injected_at_turn': perturbation_turn,
            'text': perturbation['text'][:100] + '...'
        },
        'sysar': {
            **sysar_data,
            'score': sysar_score,
            'interpretation': _interpret_single_metric(sysar_score, "SysAR")
        },
        'arco': {
            **arco_data,
            'score': arco_score,
            'interpretation': _interpret_single_metric(arco_score, "ArCo")
        },
        'overall_interpretation': interpretation,
        'keywords_used': {
            'base_keywords': base_keywords,
            'attack_keywords': attack_keywords,
            'valid_keywords': valid_keywords
        }
    }
    
    return analysis


def _interpret_single_metric(score: float, metric_name: str) -> str:
    """Provide human-readable interpretation of a single metric score."""
    if score >= 0.9:
        return f"Excellent {metric_name} - immediate response"
    elif score >= 0.7:
        return f"Very good {metric_name} - quick response"
    elif score >= 0.5:
        return f"Good {metric_name} - responded within 2 turns"
    elif score >= 0.3:
        return f"Moderate {metric_name} - slow response (3+ turns)"
    elif score > 0:
        return f"Weak {metric_name} - very slow response"
    else:
        return f"No {metric_name} - failed completely"


def _interpret_dual_metrics(sysar_score: float, arco_score: float) -> str:
    """
    Provide overall interpretation based on both metrics.
    
    Classification:
    - Perfect: SysAR=1.0, ArCo=1.0 (returned immediately with full coherence)
    - Success: SysAR>0, ArCo>0 (recovered with coherence)
    - Graceful Failure: SysAR=0, ArCo>0 (maintained coherence without full recovery)
    - Catastrophic Failure: SysAR=0, ArCo=0 (lost coherence entirely)
    """
    if sysar_score >= 0.9 and arco_score >= 0.9:
        return "✅✅ Perfect success - immediate recovery with full coherence"
    elif sysar_score > 0 and arco_score > 0:
        return "✅⚠️ Partial success - recovered with some coherence maintained"
    elif sysar_score == 0 and arco_score > 0:
        return "⚠️✅ Graceful failure - maintained coherence without full recovery to original topic"
    elif sysar_score > 0 and arco_score == 0:
        return "⚠️❌ Unstable recovery - returned to topic but lost overall coherence (rare)"
    else:  # Both zero
        return "❌❌ Catastrophic failure - lost both recovery and coherence"


def compare_dual_metrics(
    homogeneous_results: List[Dict],
    heterogeneous_results: List[Dict]
) -> Dict[str, any]:
    """
    Compare DUAL metrics (SysAR + ArCo) between homogeneous and heterogeneous debates.
    
    Args:
        homogeneous_results: List of analysis dicts for homogeneous debates
        heterogeneous_results: List of analysis dicts for heterogeneous debates
        
    Returns:
        Comparative analysis with statistics for both metrics
    """
    # Extract SysAR scores
    homo_sysar = [r['sysar']['score'] for r in homogeneous_results if r.get('sysar')]
    hetero_sysar = [r['sysar']['score'] for r in heterogeneous_results if r.get('sysar')]
    
    # Extract ArCo scores
    homo_arco = [r['arco']['score'] for r in homogeneous_results if r.get('arco')]
    hetero_arco = [r['arco']['score'] for r in heterogeneous_results if r.get('arco')]
    
    # Calculate recovery/coherence rates
    homo_recovery_rate = sum(1 for r in homogeneous_results if r.get('sysar', {}).get('recovered')) / len(homogeneous_results) if homogeneous_results else 0.0
    hetero_recovery_rate = sum(1 for r in heterogeneous_results if r.get('sysar', {}).get('recovered')) / len(heterogeneous_results) if heterogeneous_results else 0.0
    
    homo_coherence_rate = sum(1 for r in homogeneous_results if r.get('arco', {}).get('coherent')) / len(homogeneous_results) if homogeneous_results else 0.0
    hetero_coherence_rate = sum(1 for r in heterogeneous_results if r.get('arco', {}).get('coherent')) / len(heterogeneous_results) if heterogeneous_results else 0.0
    
    comparison = {
        'homogeneous': {
            'count': len(homogeneous_results),
            'sysar': {
                'avg_score': sum(homo_sysar) / len(homo_sysar) if homo_sysar else 0.0,
                'recovery_rate': homo_recovery_rate
            },
            'arco': {
                'avg_score': sum(homo_arco) / len(homo_arco) if homo_arco else 0.0,
                'coherence_rate': homo_coherence_rate
            }
        },
        'heterogeneous': {
            'count': len(heterogeneous_results),
            'sysar': {
                'avg_score': sum(hetero_sysar) / len(hetero_sysar) if hetero_sysar else 0.0,
                'recovery_rate': hetero_recovery_rate
            },
            'arco': {
                'avg_score': sum(hetero_arco) / len(hetero_arco) if hetero_arco else 0.0,
                'coherence_rate': hetero_coherence_rate
            }
        },
        'interpretation': _interpret_comparison(homo_sysar, hetero_sysar, homo_arco, hetero_arco)
    }
    
    return comparison


def _interpret_comparison(homo_sysar, hetero_sysar, homo_arco, hetero_arco) -> str:
    """Generate interpretation of comparative results."""
    homo_sysar_avg = sum(homo_sysar) / len(homo_sysar) if homo_sysar else 0.0
    hetero_sysar_avg = sum(hetero_sysar) / len(hetero_sysar) if hetero_sysar else 0.0
    homo_arco_avg = sum(homo_arco) / len(homo_arco) if homo_arco else 0.0
    hetero_arco_avg = sum(hetero_arco) / len(hetero_arco) if hetero_arco else 0.0
    
    sysar_better = hetero_sysar_avg > homo_sysar_avg
    arco_better = hetero_arco_avg > homo_arco_avg
    
    if sysar_better and arco_better:
        return "Heterogeneous systems outperform on BOTH metrics (SysAR + ArCo)"
    elif sysar_better:
        return "Heterogeneous systems show better recovery (SysAR) but similar coherence (ArCo)"
    elif arco_better:
        return "Heterogeneous systems maintain better coherence (ArCo) despite similar recovery rates (SysAR)"
    else:
        return "Homogeneous systems perform comparably or better (unexpected result)"


# Export public API
__all__ = [
    'detect_recovery_dual_metric',
    'calculate_sysar',
    'analyze_debate_sysar',
    'compare_dual_metrics',
    'BASE_KEYWORDS',
    'ATTACK_KEYWORDS'
]
