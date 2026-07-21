"""
Identity Graph (ID-RAG) - Knowledge Graph-based Identity Representation for Philosophers.

This module implements the Chronicle/Knowledge Graph structure for storing and retrieving
philosopher persona identity, following the ID-RAG (Identity Retrieval-Augmented Generation)
approach from the literature.

Architecture:
    Chronicle C_t = (V_t, E_t) where:
    - V_t: Nodes representing core identity elements (beliefs, traits, values, goals, preferences)
    - E_t: Edges representing relationships (temporal, causal, attributive, ontological)

References:
    - ID-RAG: Identity-aware RAG for persona consistency
    - KG-based Index Construction for structured knowledge representation
    - Theory of Mind (ToM) belief representation in multi-agent systems
"""

from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from datetime import datetime
import json


class NodeType(Enum):
    """Types of nodes in the philosopher's identity graph."""
    BELIEF = "belief"           # Core beliefs (e.g., "A priori knowledge is universal")
    TRAIT = "trait"             # Personal traits (e.g., "Stoicism", "Rationalism")
    VALUE = "value"             # Values (e.g., "Truth", "Justice", "Virtue")
    GOAL = "goal"               # Philosophical goals (e.g., "Understand nature of reality")
    PREFERENCE = "preference"   # Preferences (e.g., "Prefer deductive reasoning")
    CONCEPT = "concept"         # Philosophical concepts (e.g., "Categorical Imperative")
    

class EdgeType(Enum):
    """Types of relationships between identity nodes."""
    # Temporal relations
    PRECEDES = "precedes"           # One belief/concept came before another
    DERIVES_FROM = "derives_from"   # Concept derives from another
    
    # Causal relations
    CAUSES = "causes"               # One belief/value causes another
    SUPPORTS = "supports"           # One argument supports another
    CONTRADICTS = "contradicts"     # Logical contradiction
    
    # Attributive relations
    HAS_TRAIT = "has_trait"         # Philosopher has trait
    HAS_VALUE = "has_value"         # Philosopher holds value
    HAS_BELIEF = "has_belief"       # Philosopher believes proposition
    HAS_GOAL = "has_goal"           # Philosopher pursues goal
    
    # Ontological relations
    IS_A = "is_a"                   # Type hierarchy (e.g., "Virtue ethics IS_A ethical framework")
    PART_OF = "part_of"             # Composition (e.g., "Duty PART_OF deontology")
    RELATED_TO = "related_to"       # General semantic relation


class BeliefCertainty(Enum):
    """Degree of certainty for beliefs (γ parameter)."""
    CORE = 1.0          # Core beliefs - immutable, central to identity
    STRONG = 0.8        # Strong conviction
    MODERATE = 0.6      # Moderate conviction
    WEAK = 0.4          # Tentative belief
    HYPOTHETICAL = 0.2  # Hypothetical/exploratory


@dataclass
class IdentityNode:
    """
    Node in the philosopher's identity graph.
    
    Represents a core element of philosopher's persona: belief, trait, value, goal, or preference.
    Each node is grounded in source text and has metadata for retrieval and reasoning.
    """
    id: str                                 # Unique identifier
    node_type: NodeType                     # Type of identity element
    content: str                            # Natural language description
    certainty: float = 1.0                  # Degree of certainty (0-1) - γ parameter
    is_core: bool = True                    # Core belief (immutable) vs mutable
    source_chunk_id: Optional[str] = None   # Link to source text chunk
    source_text: Optional[str] = None       # Original text excerpt
    embedding: Optional[List[float]] = None # Semantic embedding for retrieval
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_triple_subject(self) -> str:
        """Convert node to subject in RDF-style triple."""
        return self.id
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize node to dictionary."""
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "content": self.content,
            "certainty": self.certainty.value if isinstance(self.certainty, BeliefCertainty) else self.certainty,
            "is_core": self.is_core,
            "source_chunk_id": self.source_chunk_id,
            "source_text": self.source_text,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class IdentityEdge:
    """
    Edge in the philosopher's identity graph.
    
    Represents a relationship between identity elements (beliefs, traits, values, etc.).
    Edges can be temporal, causal, attributive, or ontological.
    """
    source_id: str                          # Source node ID
    target_id: str                          # Target node ID
    edge_type: EdgeType                     # Type of relationship
    weight: float = 1.0                     # Strength of relationship (0-1)
    evidence: Optional[str] = None          # Textual evidence for relationship
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_triple(self) -> Tuple[str, str, str]:
        """Convert edge to RDF-style triple (subject, predicate, object)."""
        return (self.source_id, self.edge_type.value, self.target_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize edge to dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "evidence": self.evidence,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


class PhilosopherIdentityGraph:
    """
    Knowledge Graph representing philosopher's identity (Chronicle C_t).
    
    Implements ID-RAG architecture:
    - Structured identity representation using nodes (beliefs, traits, values, goals)
    - Relationship modeling via typed edges
    - Source grounding for all identity elements
    - Semantic retrieval capabilities
    - Long-horizon persona coherence
    
    The graph serves as the foundation for identity-aware response generation,
    ensuring that philosopher agents maintain consistent persona across interactions.
    """
    
    def __init__(self, philosopher_name: str, philosopher_id: str):
        """
        Initialize identity graph for a philosopher.
        
        Args:
            philosopher_name: Human-readable name (e.g., "Immanuel Kant")
            philosopher_id: Unique identifier (e.g., "kant", "mill")
        """
        self.philosopher_name = philosopher_name
        self.philosopher_id = philosopher_id
        self.graph = nx.MultiDiGraph()  # Directed graph with multiple edges allowed
        
        # Indexes for efficient retrieval
        self._nodes_by_type: Dict[NodeType, Set[str]] = {nt: set() for nt in NodeType}
        self._core_beliefs: Set[str] = set()
        self._source_chunks: Dict[str, str] = {}  # chunk_id -> text
        
        # Add root node for philosopher
        self._add_root_node()
    
    def _add_root_node(self):
        """Add root node representing the philosopher themselves."""
        root_node = IdentityNode(
            id=f"philosopher:{self.philosopher_id}",
            node_type=NodeType.CONCEPT,
            content=f"Philosopher: {self.philosopher_name}",
            certainty=1.0,
            is_core=True,
            metadata={"is_root": True}
        )
        self.add_node(root_node)
    
    def add_node(self, node: IdentityNode) -> str:
        """
        Add node to identity graph.
        
        Args:
            node: IdentityNode to add
            
        Returns:
            Node ID
        """
        self.graph.add_node(node.id, **node.to_dict())
        self._nodes_by_type[node.node_type].add(node.id)
        
        if node.is_core and node.node_type == NodeType.BELIEF:
            self._core_beliefs.add(node.id)
        
        if node.source_chunk_id and node.source_text:
            self._source_chunks[node.source_chunk_id] = node.source_text
        
        return node.id
    
    def add_edge(self, edge: IdentityEdge) -> Tuple[str, str, str]:
        """
        Add edge to identity graph.
        
        Args:
            edge: IdentityEdge to add
            
        Returns:
            Triple (source_id, edge_type, target_id)
        """
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            key=edge.edge_type.value,
            **edge.to_dict()
        )
        return edge.to_triple()
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve node by ID."""
        if node_id in self.graph.nodes:
            return dict(self.graph.nodes[node_id])
        return None
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[Dict[str, Any]]:
        """Retrieve all nodes of specific type."""
        node_ids = self._nodes_by_type[node_type]
        return [dict(self.graph.nodes[nid]) for nid in node_ids]
    
    def get_core_beliefs(self) -> List[Dict[str, Any]]:
        """Retrieve all core beliefs (immutable identity elements)."""
        return [dict(self.graph.nodes[nid]) for nid in self._core_beliefs]
    
    def get_neighbors(self, node_id: str, edge_type: Optional[EdgeType] = None) -> List[str]:
        """
        Get neighboring nodes (neighborhood expansion for context augmentation).
        
        Args:
            node_id: Node to get neighbors for
            edge_type: Optional filter by edge type
            
        Returns:
            List of neighbor node IDs
        """
        if node_id not in self.graph.nodes:
            return []
        
        neighbors = []
        for _, target, key, data in self.graph.out_edges(node_id, data=True, keys=True):
            if edge_type is None or key == edge_type.value:
                neighbors.append(target)
        
        return neighbors
    
    def get_subgraph(self, node_ids: List[str], depth: int = 1) -> nx.MultiDiGraph:
        """
        Extract subgraph around given nodes (for context augmentation).
        
        Args:
            node_ids: Starting nodes
            depth: How many hops to expand (neighborhood expansion)
            
        Returns:
            Subgraph containing relevant identity context
        """
        expanded_nodes = set(node_ids)
        
        for _ in range(depth):
            new_nodes = set()
            for node_id in expanded_nodes:
                new_nodes.update(self.get_neighbors(node_id))
            expanded_nodes.update(new_nodes)
        
        return self.graph.subgraph(expanded_nodes)
    
    def get_beliefs_about_concept(self, concept: str) -> List[Dict[str, Any]]:
        """
        Retrieve beliefs related to a specific concept.
        
        Used for semantic retrieval during debate/discussion.
        
        Args:
            concept: Concept to search for (e.g., "duty", "virtue", "happiness")
            
        Returns:
            List of belief nodes related to concept
        """
        relevant_beliefs = []
        
        for node_id in self._nodes_by_type[NodeType.BELIEF]:
            node_data = dict(self.graph.nodes[node_id])
            # Simple keyword matching (will be replaced with semantic search)
            if concept.lower() in node_data["content"].lower():
                relevant_beliefs.append(node_data)
        
        return relevant_beliefs
    
    def adaptive_search(self, strategy: dict, max_triplets: int = 25) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Search identity graph using LLM-generated strategy (ADAPTIVE ID-RAG).
        
        This implements context-aware retrieval from Platnick et al. (2025).
        Instead of static top-k, we prioritize edges and keywords based on opponent's argument.
        
        Args:
            strategy: JSON from _build_retrieval_strategy() with:
                - high_priority_relations: List[str] (e.g., ["opposes", "critiques"])
                - medium_priority_relations: List[str] (e.g., ["supports", "grounds"])
                - keywords: List[str] (e.g., ["utility", "consequentialism"])
                - focus: str (description of what to retrieve)
                - target_triplets: int (15-25)
            max_triplets: Maximum facts to return (overrides strategy if needed)
            
        Returns:
            List of (source_node_id, target_node_id, edge_data) tuples sorted by relevance
        """
        results = []
        target = min(max_triplets, strategy.get("target_triplets", 20))
        keywords = [kw.lower() for kw in strategy.get("keywords", [])]
        
        # Phase 1: HIGH-PRIORITY EDGES (opposes, critiques, contradicts, refutes)
        high_priority = strategy.get("high_priority_relations", [])
        for relation in high_priority:
            for u, v, key, data in self.graph.edges(data=True, keys=True):
                if key == relation or data.get("edge_type") == relation:
                    # Score this triplet
                    score = self._score_triplet(u, v, data, keywords, priority="high")
                    results.append((score, u, v, data))
        
        # Phase 2: MEDIUM-PRIORITY EDGES (supports, grounds, derives_from)
        if len(results) < target:
            medium_priority = strategy.get("medium_priority_relations", [])
            for relation in medium_priority:
                for u, v, key, data in self.graph.edges(data=True, keys=True):
                    if key == relation or data.get("edge_type") == relation:
                        score = self._score_triplet(u, v, data, keywords, priority="medium")
                        results.append((score, u, v, data))
        
        # Phase 3: CORE BELIEFS (always include, γ=1.0 guarantee)
        core_belief_ids = list(self._core_beliefs)[:5]  # Top 5 core beliefs
        for node_id in core_belief_ids:
            # Get outgoing edges from core beliefs
            for u, v, key, data in self.graph.out_edges(node_id, data=True, keys=True):
                score = self._score_triplet(u, v, data, keywords, priority="core")
                results.append((score, u, v, data))
        
        # Phase 4: NEIGHBORHOOD EXPANSION (r-hop from high-scoring nodes)
        if len(results) < target:
            # Get top nodes so far
            results.sort(reverse=True, key=lambda x: x[0])
            top_nodes = set([u for _, u, v, _ in results[:10]] + [v for _, u, v, _ in results[:10]])
            
            # Expand 1-hop neighborhood
            for node_id in top_nodes:
                neighbors = self.get_neighbors(node_id)
                for neighbor in neighbors:
                    # Get edges between node_id and neighbor
                    if self.graph.has_edge(node_id, neighbor):
                        # MultiDiGraph can have multiple edges between same nodes
                        edge_data_dict = self.graph.get_edge_data(node_id, neighbor)
                        if edge_data_dict:
                            for key, data in edge_data_dict.items():
                                score = self._score_triplet(node_id, neighbor, data, keywords, priority="neighbor")
                                results.append((score, node_id, neighbor, data))
        
        # Sort by score and return top results
        results.sort(reverse=True, key=lambda x: x[0])
        unique_results = []
        seen = set()
        
        for score, u, v, data in results:
            key = (u, v, data.get("edge_type", ""))
            if key not in seen and len(unique_results) < target:
                unique_results.append((u, v, data))
                seen.add(key)
        
        print(f"[OK] Adaptive search retrieved {len(unique_results)} triplets (target: {target})")
        return unique_results
    
    def _score_triplet(self, source_id: str, target_id: str, edge_data: dict, 
                       keywords: List[str], priority: str) -> float:
        """
        Score a triplet (source, edge, target) for relevance.
        
        Scoring formula:
        score = priority_weight * keyword_match_score * core_belief_boost
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_data: Edge attributes
            keywords: Keywords from strategy to match
            priority: "high", "medium", "core", or "neighbor"
            
        Returns:
            Relevance score (higher = more relevant)
        """
        # Base priority weights
        priority_weights = {
            "core": 10.0,     # Core beliefs always top priority
            "high": 5.0,      # High-priority relations (opposes, critiques)
            "medium": 2.0,    # Medium-priority relations (supports, grounds)
            "neighbor": 1.0   # Neighborhood expansion (fallback)
        }
        base_score = priority_weights.get(priority, 1.0)
        
        # Keyword matching boost
        keyword_score = 0.0
        if keywords:
            source_node = self.get_node(source_id)
            target_node = self.get_node(target_id)
            
            source_content = source_node.get("content", "").lower() if source_node else ""
            target_content = target_node.get("content", "").lower() if target_node else ""
            
            # Count keyword matches
            matches = sum(1 for kw in keywords if kw in source_content or kw in target_content)
            keyword_score = matches / len(keywords) if keywords else 0.0
        
        # Core belief boost
        core_boost = 1.5 if source_id in self._core_beliefs else 1.0
        
        # Final score
        return base_score * (1.0 + keyword_score) * core_boost
    
    def format_identity_context(self, node_ids: List[str]) -> str:
        """
        Format retrieved identity elements as natural language context.
        
        Used for context augmentation: WM'_t = WM_t ⊕ K^ID_t
        
        Args:
            node_ids: Retrieved identity nodes
            
        Returns:
            Formatted natural language string
        """
        if not node_ids:
            return ""
        
        context_parts = [f"[IDENTITY CONTEXT - {self.philosopher_name}]"]
        
        for node_id in node_ids:
            node_data = self.get_node(node_id)
            if not node_data:
                continue
            
            node_type = node_data["node_type"]
            content = node_data["content"]
            certainty = node_data["certainty"]
            is_core = node_data["is_core"]
            
            # Format based on type and certainty
            if node_type == "belief":
                prefix = "Core Belief:" if is_core else "Belief:"
                suffix = f" (certainty: {certainty:.1f})" if certainty < 1.0 else ""
                context_parts.append(f"  {prefix} {content}{suffix}")
            
            elif node_type == "value":
                context_parts.append(f"  Value: {content}")
            
            elif node_type == "trait":
                context_parts.append(f"  Trait: {content}")
            
            elif node_type == "goal":
                context_parts.append(f"  Goal: {content}")
        
        return "\n".join(context_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize entire graph to dictionary."""
        return {
            "philosopher_name": self.philosopher_name,
            "philosopher_id": self.philosopher_id,
            "nodes": [
                {
                    "id": node_id,
                    **data
                }
                for node_id, data in self.graph.nodes(data=True)
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "key": key,
                    **data
                }
                for u, v, key, data in self.graph.edges(data=True, keys=True)
            ],
            "stats": self.get_statistics()
        }
    
    def save_to_file(self, filepath: str):
        """Save graph to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'PhilosopherIdentityGraph':
        """Load graph from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        graph = cls(data["philosopher_name"], data["philosopher_id"])
        
        # Load nodes
        for node_data in data["nodes"]:
            # Skip root nodes if present
            if node_data.get("metadata", {}).get("is_root"):
                continue
            
            # Handle both 'type' (our JSON format) and 'node_type' (old format)
            node_type_str = node_data.get("type") or node_data.get("node_type")
            if not node_type_str:
                print(f"Warning: Node {node_data.get('id')} missing type field, skipping")
                continue
            
            node = IdentityNode(
                id=node_data["id"],
                node_type=NodeType(node_type_str),
                content=node_data["content"],
                certainty=node_data.get("certainty", "STRONG"),  # Default if missing
                is_core=node_data.get("is_core", False),  # Default if missing
                source_chunk_id=node_data.get("source_chunk_id"),
                source_text=node_data.get("source_text"),
                metadata=node_data.get("metadata", {}),
                created_at=datetime.fromisoformat(node_data["created_at"]) if node_data.get("created_at") else datetime.now()
            )
            graph.add_node(node)
        
        # Load edges
        for edge_data in data["edges"]:
            # Map JSON edge types to EdgeType enum
            edge_type_str = edge_data["edge_type"]
            edge_type_mapping = {
                "requires": "supports",  # Map to closest equivalent
                "is_part_of": "part_of",
                "temporal": "precedes",
                "causal": "causes",  # Map causal to causes
                # Already valid: causes, supports, contradicts, precedes
            }
            normalized_edge_type = edge_type_mapping.get(edge_type_str, edge_type_str)
            
            try:
                edge = IdentityEdge(
                    source_id=edge_data["source_id"],
                    target_id=edge_data["target_id"],
                    edge_type=EdgeType(normalized_edge_type),
                    weight=edge_data.get("weight", 1.0),
                    evidence=edge_data.get("evidence"),
                    metadata=edge_data.get("metadata", {}),
                    created_at=datetime.fromisoformat(edge_data["created_at"]) if edge_data.get("created_at") else datetime.now()
                )
                graph.add_edge(edge)
            except ValueError as e:
                print(f"Warning: Skipping edge with invalid type '{edge_type_str}': {e}")
        
        return graph
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics for monitoring and debugging."""
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "nodes_by_type": {
                nt.value: len(self._nodes_by_type[nt])
                for nt in NodeType
            },
            "core_beliefs_count": len(self._core_beliefs),
            "source_chunks_count": len(self._source_chunks),
            "density": nx.density(self.graph),
            "is_connected": nx.is_weakly_connected(self.graph)
        }
    
    def visualize_summary(self) -> str:
        """Generate human-readable summary of identity graph."""
        stats = self.get_statistics()
        
        summary = [
            f"=== Identity Graph: {self.philosopher_name} ===",
            f"Total nodes: {stats['total_nodes']}",
            f"Total edges: {stats['total_edges']}",
            "",
            "Nodes by type:"
        ]
        
        for node_type, count in stats["nodes_by_type"].items():
            if count > 0:
                summary.append(f"  - {node_type}: {count}")
        
        summary.extend([
            "",
            f"Core beliefs: {stats['core_beliefs_count']}",
            f"Source chunks: {stats['source_chunks_count']}",
            f"Graph density: {stats['density']:.3f}",
            f"Connected: {stats['is_connected']}"
        ])
        
        return "\n".join(summary)


# Example usage and testing
if __name__ == "__main__":
    # Create identity graph for Kant
    kant_graph = PhilosopherIdentityGraph("Immanuel Kant", "kant")
    
    # Add core belief
    belief1 = IdentityNode(
        id="belief:kant:categorical_imperative",
        node_type=NodeType.BELIEF,
        content="Act only according to maxims that could become universal law",
        certainty=1.0,
        is_core=True,
        source_chunk_id="groundwork_section2_para1",
        source_text="Act only according to that maxim whereby you can...",
        metadata={"work": "Groundwork of the Metaphysics of Morals", "section": 2}
    )
    kant_graph.add_node(belief1)
    
    # Add value
    value1 = IdentityNode(
        id="value:kant:duty",
        node_type=NodeType.VALUE,
        content="Duty as the supreme moral imperative",
        certainty=1.0,
        is_core=True,
        source_chunk_id="groundwork_section1_para5",
        metadata={"work": "Groundwork of the Metaphysics of Morals"}
    )
    kant_graph.add_node(value1)
    
    # Add edge: philosopher has belief
    edge1 = IdentityEdge(
        source_id="philosopher:kant",
        target_id="belief:kant:categorical_imperative",
        edge_type=EdgeType.HAS_BELIEF,
        weight=1.0,
        evidence="Central principle in Kant's moral philosophy"
    )
    kant_graph.add_edge(edge1)
    
    # Add edge: belief supports value
    edge2 = IdentityEdge(
        source_id="belief:kant:categorical_imperative",
        target_id="value:kant:duty",
        edge_type=EdgeType.SUPPORTS,
        weight=1.0
    )
    kant_graph.add_edge(edge2)
    
    # Print summary
    print(kant_graph.visualize_summary())
    print("\n" + "="*50 + "\n")
    
    # Test retrieval
    core_beliefs = kant_graph.get_core_beliefs()
    print("Core beliefs:")
    for belief in core_beliefs:
        print(f"  - {belief['content']}")
    
    print("\n" + "="*50 + "\n")
    
    # Test context formatting
    context = kant_graph.format_identity_context([belief1.id, value1.id])
    print("Identity Context:")
    print(context)
