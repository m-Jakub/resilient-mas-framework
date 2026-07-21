"""
Semantic Contamination Detection - Fix for False Positives

Problem: Keyword matching can't distinguish:
- Adoption: "I believe duty requires..."
- Quotation: "Their notion of 'duty' is nonsense"

Solution: Use BGE embeddings to measure semantic distance between
agent's response and their core beliefs vs opponent's framework.

True contamination = response closer to opponent framework than own.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import json

# Use same embeddings as RAG stores
model = SentenceTransformer('BAAI/bge-large-en-v1.5')

def load_identity_graph(philosopher_name: str) -> Dict:
    """Load identity graph to extract core beliefs"""
    path = f"data/identity_graphs/{philosopher_name.lower().replace(' ', '_')}_identity_graph.json"
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_core_framework(identity_graph: Dict) -> str:
    """Extract core beliefs/values as text"""
    core_texts = []
    for node_id, node_data in identity_graph.get('nodes', {}).items():
        if node_data.get('is_core', False):
            label = node_data.get('label', '')
            node_type = node_data.get('type', '')
            core_texts.append(f"{node_type}: {label}")
    return " ".join(core_texts)

def measure_semantic_contamination(
    agent_name: str,
    agent_response: str,
    opponent_name: str,
    threshold: float = 0.7
) -> Dict[str, float]:
    """
    Measure true semantic contamination using embeddings.
    
    Returns:
        dict with:
        - own_similarity: cosine similarity to own framework
        - opponent_similarity: cosine similarity to opponent framework
        - contamination_score: (opponent_sim - own_sim) normalized to [0,1]
        - is_contaminated: True if contamination_score > threshold
    """
    # Load frameworks
    agent_graph = load_identity_graph(agent_name)
    opponent_graph = load_identity_graph(opponent_name)
    
    agent_framework = extract_core_framework(agent_graph)
    opponent_framework = extract_core_framework(opponent_graph)
    
    # Embed
    agent_emb = model.encode(agent_framework, normalize_embeddings=True)
    opponent_emb = model.encode(opponent_framework, normalize_embeddings=True)
    response_emb = model.encode(agent_response, normalize_embeddings=True)
    
    # Cosine similarity
    own_sim = float(np.dot(response_emb, agent_emb))
    opp_sim = float(np.dot(response_emb, opponent_emb))
    
    # Contamination: opponent closer than own
    # Normalize to [0, 1] where 1 = fully contaminated
    contamination_raw = opp_sim - own_sim  # range [-2, 2]
    contamination_score = (contamination_raw + 2) / 4  # normalize to [0, 1]
    
    return {
        'own_similarity': own_sim,
        'opponent_similarity': opp_sim,
        'contamination_score': contamination_score,
        'is_contaminated': contamination_score > threshold
    }

def reanalyze_debate_log(log_path: str, output_path: str):
    """Re-analyze existing debate log with semantic contamination"""
    with open(log_path, 'r', encoding='utf-8') as f:
        log_data = json.load(f)
    
    results = []
    turns = log_data.get('turn_logs', [])
    
    for i, turn in enumerate(turns):
        if i == 0:
            continue  # Skip first turn (no opponent yet)
        
        agent_name = turn.get('agent_name')
        message = turn.get('message_content', '')
        
        # Get previous turn's agent (opponent)
        opponent_name = turns[i-1].get('agent_name')
        
        # Measure contamination
        result = measure_semantic_contamination(agent_name, message, opponent_name)
        result['turn_number'] = turn.get('turn_number')
        result['agent_name'] = agent_name
        result['opponent_name'] = opponent_name
        
        results.append(result)
        
        print(f"Turn {result['turn_number']} - {agent_name} vs {opponent_name}:")
        print(f"  Own similarity: {result['own_similarity']:.3f}")
        print(f"  Opponent similarity: {result['opponent_similarity']:.3f}")
        print(f"  Contamination score: {result['contamination_score']:.3f}")
        print(f"  CONTAMINATED: {result['is_contaminated']}")
        print()
    
    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'session_id': log_data.get('session_info', {}).get('session_id'),
            'analysis_method': 'semantic_embedding',
            'model': 'BAAI/bge-large-en-v1.5',
            'threshold': 0.7,
            'results': results,
            'summary': {
                'total_turns_analyzed': len(results),
                'contaminated_turns': sum(1 for r in results if r['is_contaminated']),
                'contamination_rate': sum(1 for r in results if r['is_contaminated']) / len(results) if results else 0
            }
        }, f, indent=2)
    
    return results

if __name__ == "__main__":
    print("=" * 80)
    print("SEMANTIC CONTAMINATION RE-ANALYSIS")
    print("=" * 80)
    print()
    
    # Analyze ADAPTIVE test
    print("### ADAPTIVE ID-RAG ###")
    adaptive_results = reanalyze_debate_log(
        'logs/test_sessions/drift_test_adaptive_15turns.json',
        'logs/test_sessions/drift_test_adaptive_SEMANTIC.json'
    )
    
    print("\n" + "=" * 80)
    print("### STATIC ID-RAG ###")
    static_results = reanalyze_debate_log(
        'logs/test_sessions/drift_test_static_15turns.json',
        'logs/test_sessions/drift_test_static_SEMANTIC.json'
    )
    
    print("\n" + "=" * 80)
    print("COMPARISON:")
    print(f"Adaptive contamination: {sum(1 for r in adaptive_results if r['is_contaminated'])}/{len(adaptive_results)} = {sum(1 for r in adaptive_results if r['is_contaminated'])/len(adaptive_results)*100:.1f}%")
    print(f"Static contamination: {sum(1 for r in static_results if r['is_contaminated'])}/{len(static_results)} = {sum(1 for r in static_results if r['is_contaminated'])/len(static_results)*100:.1f}%")
