"""
Test ID-RAG integration with PhilosopherAgent.
Creates a mini identity graph for Kant and tests retrieval in debate context.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.philosopher_agents import create_agent
from src.common.identity_graph import (
    PhilosopherIdentityGraph, 
    IdentityNode, 
    IdentityEdge,
    NodeType, 
    EdgeType, 
    BeliefCertainty
)
from langchain.schema import HumanMessage, AIMessage
from datetime import datetime


def create_kant_mini_graph() -> PhilosopherIdentityGraph:
    """Create a minimal identity graph for Kant with core deontological beliefs."""
    
    graph = PhilosopherIdentityGraph(philosopher_name="Immanuel Kant", philosopher_id="kant")
    
    # Root concept node
    philosopher_node = IdentityNode(
        id="kant_root",
        node_type=NodeType.CONCEPT,
        content="Philosopher: Immanuel Kant",
        certainty=BeliefCertainty.CORE,
        is_core=True,
        source_chunk_id="kant_bio"
    )
    graph.add_node(philosopher_node)
    
    # Core belief: Categorical Imperative
    categorical_imperative = IdentityNode(
        id="belief_categorical_imperative",
        node_type=NodeType.BELIEF,
        content="Act only according to maxims that could become universal law without contradiction",
        certainty=BeliefCertainty.CORE,
        is_core=True,
        source_chunk_id="groundwork_4:421"
    )
    graph.add_node(categorical_imperative)
    
    # Core belief: Humanity as end
    humanity_formula = IdentityNode(
        id="belief_humanity_formula",
        node_type=NodeType.BELIEF,
        content="Act in such a way that you treat humanity, whether in your own person or in the person of another, always at the same time as an end and never simply as a means",
        certainty=BeliefCertainty.CORE,
        is_core=True,
        source_chunk_id="groundwork_4:429"
    )
    graph.add_node(humanity_formula)
    
    # Core belief: Autonomy
    autonomy = IdentityNode(
        id="belief_autonomy",
        node_type=NodeType.BELIEF,
        content="Moral law must be self-legislated by rational beings through autonomous will",
        certainty=BeliefCertainty.CORE,
        is_core=True,
        source_chunk_id="groundwork_4:440"
    )
    graph.add_node(autonomy)
    
    # Value: Duty
    duty_value = IdentityNode(
        id="value_duty",
        node_type=NodeType.VALUE,
        content="Duty as the supreme moral imperative, independent of consequences",
        certainty=BeliefCertainty.CORE,
        is_core=True,
        source_chunk_id="groundwork_4:397"
    )
    graph.add_node(duty_value)
    
    # Value: Rationality
    rationality_value = IdentityNode(
        id="value_rationality",
        node_type=NodeType.VALUE,
        content="Pure practical reason as the foundation of moral judgment",
        certainty=BeliefCertainty.CORE,
        is_core=True,
        source_chunk_id="critique_practical_reason_5:3"
    )
    graph.add_node(rationality_value)
    
    # Trait: Systematic rigor
    systematic_trait = IdentityNode(
        id="trait_systematic",
        node_type=NodeType.TRAIT,
        content="Systematic and rigorous in deriving moral principles from pure reason",
        certainty=BeliefCertainty.STRONG,
        is_core=False,
        source_chunk_id="biography"
    )
    graph.add_node(systematic_trait)
    
    # Edges: Connect beliefs to values and traits
    graph.add_edge(IdentityEdge(
        source_id="kant_root",
        target_id="belief_categorical_imperative",
        edge_type=EdgeType.HAS_BELIEF,
        weight=1.0,
        evidence="Core principle in Groundwork of the Metaphysics of Morals"
    ))
    
    graph.add_edge(IdentityEdge(
        source_id="kant_root",
        target_id="belief_humanity_formula",
        edge_type=EdgeType.HAS_BELIEF,
        weight=1.0,
        evidence="Second formulation of categorical imperative"
    ))
    
    graph.add_edge(IdentityEdge(
        source_id="kant_root",
        target_id="belief_autonomy",
        edge_type=EdgeType.HAS_BELIEF,
        weight=1.0,
        evidence="Third formulation of categorical imperative"
    ))
    
    graph.add_edge(IdentityEdge(
        source_id="belief_categorical_imperative",
        target_id="value_duty",
        edge_type=EdgeType.SUPPORTS,
        weight=0.9,
        evidence="Duty derived from universalizability test"
    ))
    
    graph.add_edge(IdentityEdge(
        source_id="belief_autonomy",
        target_id="value_rationality",
        edge_type=EdgeType.SUPPORTS,
        weight=0.9,
        evidence="Autonomy grounded in pure practical reason"
    ))
    
    graph.add_edge(IdentityEdge(
        source_id="kant_root",
        target_id="value_duty",
        edge_type=EdgeType.HAS_VALUE,
        weight=1.0,
        evidence="Central value in deontological ethics"
    ))
    
    graph.add_edge(IdentityEdge(
        source_id="kant_root",
        target_id="trait_systematic",
        edge_type=EdgeType.HAS_TRAIT,
        weight=0.8,
        evidence="Characteristic approach in Critique series"
    ))
    
    return graph


def test_id_rag_retrieval():
    """Test that PhilosopherAgent can load and use identity graph."""
    
    print("="*80)
    print("TEST: ID-RAG INTEGRATION WITH PHILOSOPHER AGENT")
    print("="*80)
    
    # Step 1: Create Kant identity graph
    print("\n[Step 1] Creating Kant mini identity graph...")
    kant_graph = create_kant_mini_graph()
    stats = kant_graph.get_statistics()
    print(f"✓ Graph created: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
    print(f"✓ Core beliefs: {stats['core_beliefs_count']}")
    
    # Step 2: Create Kant agent
    print("\n[Step 2] Creating Kant agent...")
    kant_agent = create_agent("kant")
    if not kant_agent:
        print("✗ Failed to create agent")
        return
    print(f"✓ Agent created: {kant_agent.name}")
    
    # Step 3: Load identity graph into agent
    print("\n[Step 3] Loading identity graph into agent...")
    kant_agent.set_identity_graph(kant_graph)
    print("✓ Identity graph loaded")
    
    # Step 4: Test identity context retrieval
    print("\n[Step 4] Testing identity context retrieval...")
    test_query = "Should we sacrifice one person to save five in a trolley dilemma?"
    identity_context = kant_agent._retrieve_identity_context(test_query)
    
    if identity_context:
        print("✓ Identity context retrieved:")
        print("-" * 60)
        print(identity_context)
        print("-" * 60)
    else:
        print("✗ No identity context retrieved")
        return
    
    # Step 5: Test full response with ID-RAG
    print("\n[Step 5] Testing full agent response with ID-RAG...")
    chat_history = [
        HumanMessage(content="The trolley problem: A runaway trolley is headed towards five people. You can pull a lever to divert it, killing one person instead. What is the ethical course of action?")
    ]
    
    print("\n--- Agent Response (with ID-RAG) ---")
    response = kant_agent.respond(
        topic="trolley problem",
        chat_history=chat_history,
        ontology_insights=["duty", "categorical imperative"]
    )
    print(response)
    print("--- End Response ---")
    
    # Step 6: Verify identity was used
    print("\n[Step 6] Verification...")
    if any(keyword in response.lower() for keyword in ["categorical", "duty", "universal law", "maxim"]):
        print("✓ Response appears to use Kantian identity concepts")
    else:
        print("⚠ Response may not be strongly grounded in identity graph")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


def test_save_and_load():
    """Test saving and loading identity graph from file."""
    
    print("\n" + "="*80)
    print("TEST: SAVE/LOAD IDENTITY GRAPH")
    print("="*80)
    
    # Create graph
    print("\n[Step 1] Creating Kant graph...")
    kant_graph = create_kant_mini_graph()
    
    # Save to file
    filepath = "data/kant_identity_graph.json"
    print(f"\n[Step 2] Saving graph to {filepath}...")
    kant_graph.save_to_file(filepath)
    print("✓ Graph saved")
    
    # Load from file
    print(f"\n[Step 3] Loading graph from {filepath}...")
    loaded_graph = PhilosopherIdentityGraph.load_from_file(filepath)
    print("✓ Graph loaded")
    
    # Verify statistics match
    original_stats = kant_graph.get_statistics()
    loaded_stats = loaded_graph.get_statistics()
    
    print("\n[Step 4] Verifying integrity...")
    print(f"Original: {original_stats['total_nodes']} nodes, {original_stats['total_edges']} edges")
    print(f"Loaded:   {loaded_stats['total_nodes']} nodes, {loaded_stats['total_edges']} edges")
    
    if original_stats == loaded_stats:
        print("✓ Integrity verified - graphs match")
    else:
        print("✗ Integrity check failed - graphs differ")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    # Run tests
    test_id_rag_retrieval()
    print("\n\n")
    test_save_and_load()
