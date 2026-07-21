"""
Calculate ablation metrics for RQ3: Mechanism Contributions

Metrics:
- D2 (Stance Stability): Measures identity grounding (tests ID-RAG)
- A3 (Cross-Referencing): Measures strategic opponent modeling (tests ToM)
- ArCo (Coherence): Existing metric for overall resilience

Usage:
    python tests/calculate_ablation_metrics.py <transcript_path>
"""

import re
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.api_abstraction import get_philosopher_api


def calculate_d2_stance_stability(transcript_path: str) -> dict:
    """
    Measures doctrinal accuracy/framework consistency per agent.
    
    D2 = % of agent's turns that correctly use their own framework
         (without uncritically adopting opponent's framework)
    
    Errors detected:
    - Using opponent framework keywords uncritically (confusion)
    - Using no framework keywords (generic/vague)
    
    Correct usage:
    - Using own framework keywords
    - Citing opponent framework critically (in quotes, "their notion", etc.)
    
    Args:
        transcript_path: Path to debate transcript
        
    Returns:
        Dict mapping agent_name -> {'correct': int, 'total': int, 'score': float, 'errors': list}
        Also returns 'average' and 'per_team' aggregates
    """
    with open(transcript_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract turns from PHASE 3: INTER-TEAM DEBATE
    phase3_match = re.search(r'PHASE 3: INTER-TEAM DEBATE\n=+\n(.+)', content, re.DOTALL)
    if not phase3_match:
        print("Warning: Could not find PHASE 3 in transcript")
        return {'average': 0.0}
    
    phase3_content = phase3_match.group(1)
    
    # Split into individual turns: (turn_num, speaker, team, text)
    turn_pattern = r'TURN (\d+) \| (.+?) \(Team: (.+?)\)\n─+\n(.+?)(?=\n─{10,}|$)'
    turns = re.findall(turn_pattern, phase3_content, re.DOTALL)
    
    if not turns:
        print("Warning: No turns found")
        return {'average': 0.0}
    
    # Define philosopher -> expected framework mapping
    PHILOSOPHER_FRAMEWORKS = {
        'Immanuel Kant': 'deontological',
        'St. Thomas Aquinas': 'natural_law',
        'John Stuart Mill': 'utilitarian',
        'Jeremy Bentham': 'utilitarian',
        'Aristotle': 'virtue',
        'Plato': 'virtue',
        'St. Augustine': 'christian_natural_law'
    }
    
    # Framework keywords (specific, non-overlapping terms)
    FRAMEWORK_KEYWORDS = {
        'deontological': [
            'categorical imperative', 'duty', 'moral law',
            'means to an end', 'rational being', 'maxim', 'universal law',
            'kingdom of ends',
            'inherent worth of rational beings',  # Kant-specific compound
            'irrespective of circumstance',  # Kant-specific phrase
            'regardless of circumstance',
            'my doctrine recognizes'  # Kant self-reference
        ],
        'utilitarian': [
            'greatest good', 'greatest number', 'utility',
            'maximize happiness', 'minimize pain', 'calculus',
            'hedonic', 'pleasure', 'consequen', 'well-being',
            'net happiness', 'sum of happiness',
            'my principle demands',  # Utilitarian self-reference
            'principle of utility'
        ],
        'natural_law': [
            'natural law', 'intrinsically evil',
            'right reason', 'immutable principle', 'divine law',
            'thomistic', 'aquinas',
            'accessible through reason',  # Aquinas-specific
            'inherent nature of the act'
        ],
        'virtue': [
            'virtue', 'character', 'prudence', 'justice',
            'soul', 'moral development', 'flourishing', 'eudaimonia',
            'phronesis', 'golden mean', 'cultivation',
            'practical wisdom'  # Aristotle-specific
        ],
        'christian_natural_law': [
            'grace', 'original sin', 'love of god', 'divine authority',
            'charity', 'faith', 'will of god', 'corruption of soul',
            'augustinian', 'city of god'
        ]
    }
    
    # Shared philosophical terms that don't indicate framework confusion
    # These are used across multiple traditions and should be ignored
    SHARED_KEYWORDS = [
        'intent', 'intention', 'intentionally', 'intend',
        'inherent', 'inherently', 'intrinsic', 'intrinsically',
        'principle', 'principles',
        'moral', 'morally', 'ethical', 'ethically',
        'wrong', 'right', 'good', 'bad', 'evil',
        'harm', 'benefit', 'outcome', 'action', 'act',
        'dignity', 'worth', 'value'
    ]
    
    # Critical citation patterns (opponent framework mentioned critically)
    CRITICAL_PATTERNS = [
        r'"[^"]+"',  # Anything in quotes
        r'their (notion|doctrine|principle|framework|claim|assertion|position|view)',
        r'esteemed \w+\'s',
        r'opponent\'s',
        r'supposedly',
        r'so-called',
        r'purported',
        r'alleged'
    ]
    
    # Anti-utilitarian phrases (deontological/natural law critiques of consequentialism)
    ANTI_UTILITARIAN_PHRASES = [
        'irrespective of consequences',
        'regardless of consequences',
        'regardless of outcome',
        'regardless of their perceived outcome',
        'not determined by consequences',
        'not solely determined by consequences',
        'ignore consequences',
        'ignores consequences',
        'disregard consequences',
        'ends do not justify',
        'consequences are not the only',
        'focus solely on consequences',
        'focused solely on consequences'
    ]
    
    # Anti-deontological phrases (utilitarian/pragmatic critiques of duty ethics)
    ANTI_DEONTOLOGICAL_PHRASES = [
        'rigid adherence',
        'blindly following',
        'abstract principle',
        'empty formalism',
        'offers no practical guidance',
        'provides no mechanism',
        'lacks practical application',
        'impractical',
        'inflexible'
    ]
    
    # Group turns by agent
    agent_data = {}
    for turn_num, speaker, team, text in turns:
        if speaker not in agent_data:
            agent_data[speaker] = {
                'turns': [],
                'team': team,
                'correct': 0,
                'errors': []
            }
        agent_data[speaker]['turns'].append((int(turn_num), text))
    
    # Analyze each agent
    agent_d2_scores = {}
    
    for agent, data in agent_data.items():
        if agent not in PHILOSOPHER_FRAMEWORKS:
            print(f"  [{agent}]: Unknown philosopher, skipping")
            continue
        
        expected_framework = PHILOSOPHER_FRAMEWORKS[agent]
        own_keywords = FRAMEWORK_KEYWORDS[expected_framework]
        
        # Get opponent frameworks (all others)
        opponent_frameworks = {k: v for k, v in FRAMEWORK_KEYWORDS.items() 
                              if k != expected_framework}
        
        print(f"\n  [{agent}] Expected framework: {expected_framework}")
        
        correct_turns = 0
        errors = []
        
        for turn_idx, turn_text in data['turns']:
            turn_lower = turn_text.lower()
            
            # Check if turn uses own framework
            own_score = sum(1 for kw in own_keywords if re.search(r'\b' + re.escape(kw) + r'\b', turn_lower))
            
            # Check for opponent framework keywords (uncritical use)
            opponent_uncritical_uses = []
            for opp_framework, opp_keywords in opponent_frameworks.items():
                for kw in opp_keywords:
                    # Use word boundary matching to avoid substring false positives
                    kw_pattern = r'\b' + re.escape(kw) + r'\b'
                    if re.search(kw_pattern, turn_lower):
                        # Skip if it's a shared keyword (used across multiple frameworks)
                        if any(shared_kw in kw or kw in shared_kw for shared_kw in SHARED_KEYWORDS):
                            continue
                        
                        is_critical = False
                        
                        # Check 1: General critical patterns
                        if any(re.search(pattern, turn_text, re.IGNORECASE) for pattern in CRITICAL_PATTERNS):
                            is_critical = True
                        
                        # Check 2: Context window around keyword
                        if not is_critical:
                            context_window = 50
                            kw_pos = turn_lower.find(kw)
                            if kw_pos != -1:
                                context = turn_lower[max(0, kw_pos-context_window):min(len(turn_lower), kw_pos+len(kw)+context_window)]
                                if any(marker in context for marker in ['their', 'opponent', '"', 'esteemed', 'supposedly']):
                                    is_critical = True
                        
                        # Check 3: Framework-specific anti-phrases
                        if not is_critical:
                            if opp_framework == 'utilitarian':
                                # Deontologist/natural law critiquing consequentialism
                                if any(anti_phrase in turn_lower for anti_phrase in ANTI_UTILITARIAN_PHRASES):
                                    is_critical = True
                            elif opp_framework == 'deontological':
                                # Utilitarian critiquing duty ethics
                                if any(anti_phrase in turn_lower for anti_phrase in ANTI_DEONTOLOGICAL_PHRASES):
                                    is_critical = True
                        
                        if not is_critical:
                            opponent_uncritical_uses.append((opp_framework, kw))
            
            # Classify turn
            if opponent_uncritical_uses:
                error_msg = f"Turn {turn_idx}: Used {opponent_uncritical_uses[0][0]} keyword '{opponent_uncritical_uses[0][1]}' uncritically"
                errors.append(error_msg)
                print(f"    Turn {turn_idx}: ✗ ERROR - Framework confusion ({opponent_uncritical_uses[0][1]})")
            elif own_score == 0:
                error_msg = f"Turn {turn_idx}: No framework keywords detected"
                errors.append(error_msg)
                print(f"    Turn {turn_idx}: ✗ ERROR - Generic/vague")
            else:
                correct_turns += 1
                print(f"    Turn {turn_idx}: ✓ Correct (own framework, {own_score} keywords)")
        
        total_turns = len(data['turns'])
        d2_score = correct_turns / total_turns if total_turns > 0 else 0.0
        
        agent_d2_scores[agent] = {
            'correct': correct_turns,
            'total': total_turns,
            'score': d2_score,
            'errors': errors[:3],  # Keep first 3 errors
            'team': data['team']
        }
        
        print(f"    → D2 for {agent}: {d2_score:.2f} ({correct_turns}/{total_turns} correct)")
        if errors:
            print(f"       Sample errors: {errors[0]}")
    
    # Calculate averages
    if agent_d2_scores:
        scores_only = [v['score'] for v in agent_d2_scores.values()]
        avg_d2 = sum(scores_only) / len(scores_only)
        
        # Per-team aggregates
        team_scores = {}
        for agent, data in agent_d2_scores.items():
            team = data['team']
            if team not in team_scores:
                team_scores[team] = []
            team_scores[team].append(data['score'])
        
        team_averages = {team: sum(scores)/len(scores) for team, scores in team_scores.items()}
        
        result = {
            'per_agent': agent_d2_scores,
            'average': avg_d2,
            'per_team': team_averages
        }
        
        print(f"\n  → OVERALL AVERAGE D2: {avg_d2:.2f}")
        print(f"  → PER-TEAM AVERAGES:")
        for team, avg in team_averages.items():
            print(f"     {team}: {avg:.2f}")
        
        return result
    else:
        return {'average': 0.0, 'per_agent': {}, 'per_team': {}}


def extract_stance(turns: list[str]) -> str:
    """
    Extract agent's stance from turn text.
    
    Returns:
        'pull_lever', 'not_pull_lever', or 'ambiguous'
    """
    text = ' '.join(turns).lower()
    
    # Keyword-based heuristics
    pull_keywords = [
        'i would pull', 'should pull', 'must pull', 'ought to pull',
        'pulling the lever', 'divert the trolley', 'redirect',
        'sacrifice one', 'save five', 'greatest good',
        'utilitarian', 'maximize happiness', 'minimize harm',
        'five lives outweigh', 'arithmetic', 'calculus'
    ]
    
    not_pull_keywords = [
        'i would not pull', 'should not pull', 'cannot pull', 'must not pull',
        'refrain from', 'do not act', 'not intervene', 'inaction',
        'categorical imperative', 'means to an end', 'use a person',
        'intrinsically evil', 'natural law', 'dignity',
        'directly cause harm', 'intentionally kill', 'sacred duty'
    ]
    
    pull_score = sum(1 for kw in pull_keywords if kw in text)
    not_pull_score = sum(1 for kw in not_pull_keywords if kw in text)
    
    # Require clear majority (at least 2:1 ratio)
    if pull_score > not_pull_score * 2:
        return 'pull_lever'
    elif not_pull_score > pull_score * 2:
        return 'not_pull_lever'
    elif pull_score > not_pull_score:
        return 'pull_lever'
    elif not_pull_score > pull_score:
        return 'not_pull_lever'
    else:
        # Fallback: LLM classification for ambiguous cases
        return llm_classify_stance(text)


def llm_classify_stance(text: str) -> str:
    """
    LLM-based fallback for ambiguous stance detection.
    Uses existing api_abstraction layer.
    """
    # Truncate to avoid token limits
    truncated_text = text[:1000]
    
    prompt = f"""Analyze this philosophical argument and determine the agent's stance on the trolley problem.

Text: "{truncated_text}"

Does this agent advocate for:
A) Pulling the lever (sacrificing one to save five)
B) Not pulling the lever (refusing to directly cause harm)

Respond with ONLY one word: "pull_lever" or "not_pull_lever"
"""
    
    try:
        api = get_philosopher_api()
        llm = api.get_llm()
        response = llm.invoke(prompt)
        
        # Handle different response types (LangChain returns different objects)
        if hasattr(response, 'content'):
            result = response.content.strip().lower()
        else:
            result = str(response).strip().lower()
        
        if 'pull_lever' in result:
            return 'pull_lever'
        elif 'not_pull' in result or 'not_pull_lever' in result:
            return 'not_pull_lever'
        else:
            return 'ambiguous'
    except Exception as e:
        print(f"  Warning: LLM classification failed: {e}")
        return 'ambiguous'


def calculate_a3_cross_referencing(transcript_path: str) -> float:
    """
    Measures explicit opponent argument engagement.
    
    Counts turns with explicit references to opponent's arguments.
    
    Args:
        transcript_path: Path to debate transcript
        
    Returns:
        Float 0-1: Proportion of turns with cross-references
    """
    with open(transcript_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract turns from PHASE 3: INTER-TEAM DEBATE
    phase3_match = re.search(r'PHASE 3: INTER-TEAM DEBATE\n=+\n(.+)', content, re.DOTALL)
    if not phase3_match:
        print("Warning: Could not find PHASE 3 in transcript")
        return 0.0
    
    phase3_content = phase3_match.group(1)
    
    # Split into individual turns
    turn_pattern = r'TURN (\d+) \| (.+?) \(Team: (.+?)\)\n─+\n(.+?)(?=\n─{10,}|$)'
    turns = re.findall(turn_pattern, phase3_content, re.DOTALL)
    
    if not turns:
        print("Warning: No turns found")
        return 0.0
    
    # Cross-reference patterns (case-insensitive)
    cross_ref_patterns = [
        # Direct address (you/your)
        r'\byou (say|said|claim|argue|suggest|propose|contend|assert|mention|state|believe)',
        r'\byour (argument|position|view|stance|claim|reasoning|point|assertion|contention)',
        r'\byou\'ve (mentioned|stated|argued|claimed|said)',
        r'\bas you (pointed out|mentioned|stated|argued|said)',
        r'\bwhile you (focus|emphasize|prioritize|consider)',
        r'\byour view that',
        r'\baccording to you',
        r'\byou would have us',
        
        # Third person references (their/opponent)
        r'\btheir (argument|position|view|stance|claim|reasoning|notion|doctrine|assertion)',
        r'\bopponent(?:\'s)? (argument|position|view|claim|reasoning)',
        r'\bthe esteemed (Mill|Bentham|Kant|Aquinas|Aristotle|Augustine|Plato)',
        r'\bmy (colleague|opponent|adversary)(?:\'s)?',
        
        # Named philosopher references
        r'\b(Mill|Bentham|Kant|Aquinas|Aristotle|Augustine|Plato)(?:\'s)? (says|said|argues|argued|claims|claimed|suggests|proposed|assertion|position|view)',
        
        # Explicit response markers
        r'\bin response to',
        r'\baddressing (the|their|your)',
        r'\bregarding (the|their|your)',
        r'\bthis (assertion|claim|argument|position)',
    ]
    
    turns_with_cross_ref = 0
    
    for turn_num, speaker, team, turn_text in turns:
        turn_lower = turn_text.lower()
        has_cross_ref = any(re.search(pattern, turn_lower) for pattern in cross_ref_patterns)
        
        if has_cross_ref:
            turns_with_cross_ref += 1
            print(f"  Turn {turn_num} ({speaker}): ✓ Cross-reference detected")
        else:
            print(f"  Turn {turn_num} ({speaker}): ✗ No cross-reference")
    
    a3_score = turns_with_cross_ref / len(turns)
    print(f"  → A3 Cross-Referencing: {a3_score:.2f} ({turns_with_cross_ref}/{len(turns)} turns)")
    
    return a3_score


def calculate_arco(transcript_path: str) -> float:
    """
    Calculate Argumentative Coherence (ArCo) using existing keyword-based approach.
    
    This is a simplified implementation. For full SysAR calculation, use analyze_sysar_log.py
    
    Returns:
        Float 0-1: Proportion of coherent turns (on-topic)
    """
    with open(transcript_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract turns from PHASE 3
    phase3_match = re.search(r'PHASE 3: INTER-TEAM DEBATE\n=+\n(.+)', content, re.DOTALL)
    if not phase3_match:
        return 0.0
    
    phase3_content = phase3_match.group(1)
    turn_pattern = r'TURN (\d+) \| (.+?) \(Team: (.+?)\)\n─+\n(.+?)(?=\n─{10,}|$)'
    turns = re.findall(turn_pattern, phase3_content, re.DOTALL)
    
    if not turns:
        return 0.0
    
    # Base topic keywords (trolley problem)
    base_keywords = [
        'trolley', 'lever', 'track', 'dilemma', 'choice', 'decision',
        'life', 'lives', 'person', 'people', 'death', 'harm',
        'utilitarian', 'deontolog', 'virtue', 'consequen', 'duty',
        'moral', 'ethical', 'principle', 'right', 'wrong'
    ]
    
    coherent_turns = 0
    for turn_num, speaker, team, turn_text in turns:
        turn_lower = turn_text.lower()
        if any(keyword in turn_lower for keyword in base_keywords):
            coherent_turns += 1
    
    arco_score = coherent_turns / len(turns)
    print(f"  → ArCo (Coherence): {arco_score:.2f} ({coherent_turns}/{len(turns)} coherent turns)")
    
    return arco_score


def calculate_ablation_metrics(transcript_path: str) -> dict:
    """
    Calculate all ablation metrics for a transcript.
    
    Returns:
        {
            'd2_stance_stability': float,
            'a3_cross_referencing': float,
            'arco': float
        }
    """
    print(f"\n{'='*80}")
    print(f"ABLATION METRICS: {Path(transcript_path).name}")
    print(f"{'='*80}\n")
    
    print("📊 D2: STANCE STABILITY (tests ID-RAG)")
    print("-" * 80)
    d2 = calculate_d2_stance_stability(transcript_path)
    
    print("\n🔗 A3: CROSS-REFERENCING (tests ToM)")
    print("-" * 80)
    a3 = calculate_a3_cross_referencing(transcript_path)
    
    print("\n📈 ArCo: ARGUMENTATIVE COHERENCE (overall resilience)")
    print("-" * 80)
    arco = calculate_arco(transcript_path)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    # D2 is now a dict with detailed structure
    if isinstance(d2, dict) and 'per_agent' in d2:
        print("D2 (Doctrinal Accuracy) per-agent:")
        for agent, data in d2.get('per_agent', {}).items():
            score = data['score']
            correct = data['correct']
            total = data['total']
            print(f"  {agent}: {score:.2f} ({correct}/{total})")
        
        if 'average' in d2:
            print(f"\nAverage D2: {d2['average']:.2f}")
        
        if 'per_team' in d2 and d2['per_team']:
            print(f"\nPer-team D2:")
            for team, score in d2['per_team'].items():
                print(f"  {team}: {score:.2f}")
    else:
        # Fallback for simple dict or float
        if isinstance(d2, dict) and 'average' in d2:
            print(f"D2 (Doctrinal Accuracy): {d2['average']:.2f}")
        elif isinstance(d2, (int, float)):
            print(f"D2 (Doctrinal Accuracy): {d2:.2f}")
        else:
            print(f"D2 (Doctrinal Accuracy): {d2}")
    
    print(f"A3 (Cross-Referencing):  {a3:.2f}")
    print(f"ArCo (Coherence):        {arco:.2f}")
    print(f"{'='*80}\n")
    
    return {
        'd2_stance_stability': d2,
        'a3_cross_referencing': a3,
        'arco': arco
    }


if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Calculate ablation metrics for debate transcripts')
    parser.add_argument('transcript_path', help='Path to the debate transcript file')
    parser.add_argument('--output-json', action='store_true', 
                       help='Save metrics to JSON file (transcript_path_metrics.json)')
    
    args = parser.parse_args()
    
    if not Path(args.transcript_path).exists():
        print(f"Error: File not found: {args.transcript_path}")
        sys.exit(1)
    
    metrics = calculate_ablation_metrics(args.transcript_path)
    
    # Save to JSON if requested
    if args.output_json:
        output_path = Path(args.transcript_path).with_suffix('') / '_metrics.json'
        # Fix path construction
        output_path = str(args.transcript_path).replace('.txt', '_metrics.json')
        
        # Prepare JSON-serializable version
        json_metrics = {
            'transcript': args.transcript_path,
            'd2_doctrinal_accuracy': metrics['d2_stance_stability'],
            'a3_cross_referencing': metrics['a3_cross_referencing'],
            'arco_coherence': metrics['arco']
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_metrics, f, indent=2)
        
        print(f"\n💾 Metrics saved to: {output_path}")

