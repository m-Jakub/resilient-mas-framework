"""
Conceptual Coverage Score (CCS)

Measures the breadth of ethical concepts introduced during a debate.
Higher coverage = more diverse philosophical perspectives explored.

Metric Definition:
    CCS = |unique_concepts_mentioned| / |total_ethical_vocabulary|

Where:
- unique_concepts_mentioned: Set of ethical terms/concepts used in debate
- total_ethical_vocabulary: Union of all concepts from participating philosophers

Expected Results:
- Baseline (single LLM): LOW (one perspective)
- Homogeneous (same school): MEDIUM (similar frameworks)
- Heterogeneous (different schools): HIGH (diverse frameworks)
- Full Heterogeneous (max diversity): HIGHEST (broadest coverage)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Set, Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

# Comprehensive ethical vocabulary by philosophical school
ETHICAL_VOCABULARY = {
    # Virtue Ethics (Plato, Aristotle, Aquinas, Augustine)
    "virtue_ethics": [
        "virtue", "vice", "character", "excellence", "eudaimonia", "flourishing",
        "practical wisdom", "prudence", "courage", "temperance", "justice",
        "wisdom", "good life", "telos", "purpose", "habit", "habituation",
        "mean", "golden mean", "excess", "deficiency", "moral education",
        "phronesis", "arete", "ethos", "sophia", "nous", "reason",
        # Aquinas-specific
        "natural law", "divine law", "eternal law", "human law", "common good",
        "grace", "sin", "theological virtues", "cardinal virtues", "beatitude",
        # Augustine-specific
        "city of god", "original sin", "free will", "predestination", "faith",
        "charity", "love of god", "amor dei", "cupiditas", "caritas"
    ],
    
    # Deontology (Kant)
    "deontology": [
        "duty", "obligation", "categorical imperative", "maxim", "universal law",
        "good will", "dignity", "respect", "autonomy", "rational agency",
        "means", "end", "kingdom of ends", "moral law", "practical reason",
        "imperative", "hypothetical imperative", "perfect duty", "imperfect duty",
        "humanity formula", "universalizability", "rational nature", "freedom",
        "noumena", "phenomena", "pure reason", "moral worth", "inclination"
    ],
    
    # Utilitarianism (Mill, Bentham)
    "utilitarianism": [
        "utility", "pleasure", "pain", "happiness", "greatest happiness",
        "greatest number", "consequence", "calculus", "hedonism", "felicific calculus",
        "intensity", "duration", "certainty", "propinquity", "fecundity", "purity",
        "extent", "higher pleasures", "lower pleasures", "quality", "quantity",
        "harm principle", "liberty", "individual rights", "general welfare",
        "public good", "aggregate happiness", "maximization", "minimize suffering"
    ],
    
    # Nietzsche (Existential/Perspectivism)
    "nietzschean": [
        "will to power", "master morality", "slave morality", "ressentiment",
        "revaluation of values", "beyond good and evil", "genealogy", "ascetic ideal",
        "herd morality", "noble", "base", "strength", "weakness", "life-affirming",
        "life-denying", "amor fati", "eternal recurrence", "overman", "ubermensch",
        "nihilism", "perspectivism", "critique", "decadence", "instinct", "passion"
    ],
    
    # Cross-cutting concepts (used by multiple schools)
    "general": [
        "morality", "ethics", "right", "wrong", "good", "evil", "ought", "should",
        "action", "agent", "patient", "intention", "motive", "responsibility",
        "blame", "praise", "permissible", "forbidden", "supererogatory",
        "dilemma", "conflict", "resolution", "principle", "rule", "value",
        "norm", "judgment", "reasoning", "justification", "legitimacy"
    ]
}


def get_philosopher_vocabulary(philosopher_key: str) -> Set[str]:
    """
    Get the expected vocabulary for a specific philosopher.
    
    Args:
        philosopher_key: e.g., 'kant', 'mill', 'plato', 'nietzsche'
    
    Returns:
        Set of ethical concepts this philosopher is expected to use
    """
    school_mapping = {
        "kant": "deontology",
        "mill": "utilitarianism",
        "bentham": "utilitarianism",
        "plato": "virtue_ethics",
        "aristotle": "virtue_ethics",
        "aquinas": "virtue_ethics",
        "st_augustine": "virtue_ethics",
        "nietzsche": "nietzschean"
    }
    
    school = school_mapping.get(philosopher_key, "general")
    return set(ETHICAL_VOCABULARY.get(school, [])) | set(ETHICAL_VOCABULARY["general"])


def extract_concepts_from_text(text: str, vocabulary: Set[str]) -> Set[str]:
    """
    Extract ethical concepts from text based on vocabulary.
    
    Uses simple word/phrase matching (case-insensitive).
    
    Args:
        text: Agent's response text
        vocabulary: Set of concepts to search for
    
    Returns:
        Set of concepts found in text
    """
    text_lower = text.lower()
    found_concepts = set()
    
    for concept in vocabulary:
        # Check for whole word/phrase match
        # Simple version: substring search
        if concept.lower() in text_lower:
            found_concepts.add(concept)
    
    return found_concepts


def compute_conceptual_coverage(log_file: Path) -> Dict:
    """
    Compute Conceptual Coverage Score for a debate log.
    
    Args:
        log_file: Path to debate JSON log
    
    Returns:
        Dict with coverage statistics
    """
    print("\n" + "="*80)
    print("CONCEPTUAL COVERAGE ANALYSIS")
    print("="*80 + "\n")
    
    with open(log_file, 'r', encoding='utf-8') as f:
        debate_log = json.load(f)
    
    turns = debate_log.get('turn_logs', [])
    
    # Identify participating philosophers
    participants = set()
    for turn in turns:
        agent_id = turn.get('agent_id', '')
        if agent_id:
            participants.add(agent_id)
    
    print(f"Participants: {', '.join(participants)}")
    print(f"Total turns: {len(turns)}\n")
    
    # Build total vocabulary (union of all participants' expected concepts)
    total_vocabulary = set()
    for agent_id in participants:
        agent_vocab = get_philosopher_vocabulary(agent_id)
        total_vocabulary.update(agent_vocab)
    
    print(f"Total ethical vocabulary size: {len(total_vocabulary)} concepts")
    
    # Extract concepts used in debate
    concepts_used = set()
    concepts_per_turn = []
    
    for turn in turns:
        message = turn.get('message_content', '')
        turn_concepts = extract_concepts_from_text(message, total_vocabulary)
        concepts_used.update(turn_concepts)
        concepts_per_turn.append({
            'turn': turn.get('turn_number'),
            'agent': turn.get('agent_name'),
            'concepts': list(turn_concepts),
            'count': len(turn_concepts)
        })
    
    # Compute coverage score
    coverage_score = len(concepts_used) / len(total_vocabulary) if total_vocabulary else 0
    
    print(f"\nConcepts actually used: {len(concepts_used)}")
    print(f"Coverage Score: {coverage_score:.2%}\n")
    
    # Show top concepts by frequency
    concept_frequency = defaultdict(int)
    for turn_data in concepts_per_turn:
        for concept in turn_data['concepts']:
            concept_frequency[concept] += 1
    
    top_concepts = sorted(concept_frequency.items(), key=lambda x: x[1], reverse=True)[:20]
    
    print("="*80)
    print("TOP 20 CONCEPTS BY FREQUENCY")
    print("="*80)
    for concept, freq in top_concepts:
        print(f"  {concept:30s} - {freq} mentions")
    
    # Per-agent concept usage
    print("\n" + "="*80)
    print("CONCEPT USAGE PER AGENT")
    print("="*80)
    
    agent_concepts = defaultdict(set)
    for turn in turns:
        agent_id = turn.get('agent_id', '')
        message = turn.get('message_content', '')
        turn_concepts = extract_concepts_from_text(message, total_vocabulary)
        agent_concepts[agent_id].update(turn_concepts)
    
    for agent_id in sorted(participants):
        agent_name = next((t['agent_name'] for t in turns if t['agent_id'] == agent_id), agent_id)
        concepts = agent_concepts[agent_id]
        expected_vocab = get_philosopher_vocabulary(agent_id)
        coverage = len(concepts) / len(expected_vocab) if expected_vocab else 0
        
        print(f"\n[{agent_name}]")
        print(f"  Concepts used: {len(concepts)}")
        print(f"  Expected vocabulary: {len(expected_vocab)}")
        print(f"  Personal coverage: {coverage:.1%}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return {
        "coverage_score": coverage_score,
        "total_vocabulary_size": len(total_vocabulary),
        "concepts_used_count": len(concepts_used),
        "concepts_used": sorted(list(concepts_used)),
        "top_concepts": top_concepts[:20],
        "agent_coverage": {
            agent_id: {
                "count": len(agent_concepts[agent_id]),
                "concepts": sorted(list(agent_concepts[agent_id]))
            }
            for agent_id in participants
        }
    }


if __name__ == "__main__":
    # Test on existing logs
    log_dir = Path("logs/test_sessions")
    
    # Find latest drift test logs
    test_logs = sorted(log_dir.glob("drift_test_*_15turns.json"))
    
    if test_logs:
        print("Available test logs:")
        for i, log in enumerate(test_logs, 1):
            print(f"  {i}. {log.name}")
        
        # Analyze first log as example
        print("\n" + "="*80)
        print(f"ANALYZING: {test_logs[0].name}")
        print("="*80)
        
        results = compute_conceptual_coverage(test_logs[0])
        
        # Save results
        output_file = log_dir / f"conceptual_coverage_{test_logs[0].stem}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_file}")
    else:
        print("No test logs found in logs/test_sessions/")
