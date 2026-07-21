"""
Enhanced ontology with team-level concept tracking and analysis.
Extends the base ontology with richer semantic relationships.
"""

from typing import Dict, List, Set
from src.common.ontology import ETHICAL_CONCEPTS


class EnhancedOntology:
    """
    Enhanced ontology for ethical concepts with relationship tracking.
    Integrates with team memory to track concept usage across debates.
    """
    
    def __init__(self):
        self.base_concepts = ETHICAL_CONCEPTS
        self.concept_relationships = self._build_relationships()
        self.concept_usage_history: List[Dict] = []
    
    def _build_relationships(self) -> Dict[str, Dict]:
        """
        Build relationships between ethical concepts.
        Shows which concepts are related, opposed, or complementary.
        """
        return {
            "consequence": {
                "related": ["utility"],
                "opposed": ["duty", "categorical imperative"],
                "school": "utilitarianism",
                "weight": "primary"
            },
            "utility": {
                "related": ["consequence"],
                "opposed": ["duty"],
                "school": "utilitarianism",
                "weight": "primary"
            },
            "duty": {
                "related": ["categorical imperative"],
                "opposed": ["consequence", "utility"],
                "school": "deontology",
                "weight": "primary"
            },
            "categorical imperative": {
                "related": ["duty"],
                "opposed": ["consequence", "utility"],
                "school": "deontology",
                "weight": "primary"
            },
            "virtue": {
                "related": ["natural law", "intention"],
                "opposed": [],
                "school": "christian_ethics",
                "weight": "primary"
            },
            "natural law": {
                "related": ["virtue", "intention"],
                "opposed": [],
                "school": "christian_ethics",
                "weight": "primary"
            },
            "intention": {
                "related": ["virtue", "natural law", "double effect"],
                "opposed": ["consequence"],
                "school": "christian_ethics",
                "weight": "secondary"
            },
            "double effect": {
                "related": ["intention"],
                "opposed": [],
                "school": "christian_ethics",
                "weight": "secondary"
            }
        }
    
    def get_concept_info(self, concept_key: str) -> Dict:
        """Get full information about a concept."""
        if concept_key not in self.base_concepts:
            return {}
        
        base_info = self.base_concepts[concept_key]
        relations = self.concept_relationships.get(concept_key, {})
        
        return {
            "key": concept_key,
            "definition": base_info["definition"],
            "primary_school": base_info["primary_school"],
            "related_concepts": relations.get("related", []),
            "opposed_concepts": relations.get("opposed", []),
            "weight": relations.get("weight", "secondary")
        }
    
    def get_school_concepts(self, school: str) -> List[str]:
        """Get all concepts associated with a philosophical school."""
        return [
            key for key, info in self.base_concepts.items()
            if info["primary_school"] == school
        ]
    
    def analyze_concept_conflict(self, concepts: List[str]) -> Dict:
        """
        Analyze potential conflicts between concepts used in debate.
        
        Returns:
            Dictionary with conflict analysis
        """
        conflicts = []
        schools_involved = set()
        
        for concept in concepts:
            if concept not in self.concept_relationships:
                continue
            
            relations = self.concept_relationships[concept]
            schools_involved.add(relations.get("school"))
            
            # Check for opposed concepts
            opposed = relations.get("opposed", [])
            for opp in opposed:
                if opp in concepts:
                    conflicts.append({
                        "concept1": concept,
                        "concept2": opp,
                        "type": "opposition"
                    })
        
        return {
            "total_concepts": len(concepts),
            "schools_involved": list(schools_involved),
            "conflicts": conflicts,
            "conflict_count": len(conflicts)
        }
    
    def suggest_complementary_concepts(self, concept: str) -> List[str]:
        """Suggest related concepts that might enrich the discussion."""
        if concept not in self.concept_relationships:
            return []
        
        return self.concept_relationships[concept].get("related", [])
    
    def track_concept_usage(self, concept: str, team_id: str, turn: int):
        """Track when and by whom a concept was used."""
        self.concept_usage_history.append({
            "concept": concept,
            "team_id": team_id,
            "turn": turn
        })
    
    def get_usage_summary(self) -> Dict:
        """Get summary of concept usage across debate."""
        if not self.concept_usage_history:
            return {"total_uses": 0, "concepts_used": []}
        
        concepts_used = list(set(item["concept"] for item in self.concept_usage_history))
        teams_active = list(set(item["team_id"] for item in self.concept_usage_history))
        
        # Count usage per concept
        concept_counts = {}
        for item in self.concept_usage_history:
            concept = item["concept"]
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
        
        return {
            "total_uses": len(self.concept_usage_history),
            "concepts_used": concepts_used,
            "unique_concepts": len(concepts_used),
            "teams_active": teams_active,
            "concept_frequency": concept_counts,
            "most_used": max(concept_counts.items(), key=lambda x: x[1])[0] if concept_counts else None
        }
    
    def generate_debate_insights(self, team_memories: Dict) -> str:
        """
        Generate insights about the debate based on concept usage and team memories.
        
        Args:
            team_memories: Dict of team_id -> TeamMemory
            
        Returns:
            Formatted string with debate insights
        """
        insights = []
        
        # Analyze concepts used
        all_concepts = set()
        for memory in team_memories.values():
            all_concepts.update(memory.ethical_concepts)
        
        if all_concepts:
            insights.append(f"Ethical concepts discussed: {', '.join(all_concepts)}")
            
            # Identify philosophical schools involved
            schools = set()
            for concept in all_concepts:
                if concept in self.base_concepts:
                    schools.add(self.base_concepts[concept]["primary_school"])
            
            insights.append(f"Philosophical schools: {', '.join(schools)}")
            
            # Check for conflicts
            conflict_analysis = self.analyze_concept_conflict(list(all_concepts))
            if conflict_analysis["conflicts"]:
                insights.append(f"Conceptual tensions identified: {conflict_analysis['conflict_count']}")
        
        return "\n".join(insights) if insights else "No conceptual analysis available."


# Global enhanced ontology instance
ENHANCED_ONTOLOGY = EnhancedOntology()


def integrate_ontology_with_team(team, ontology: EnhancedOntology = None):
    """
    Integrate ontology tracking with team memory.
    
    Args:
        team: Team instance
        ontology: EnhancedOntology instance (optional)
    """
    if ontology is None:
        ontology = ENHANCED_ONTOLOGY
    
    # Add ontology reference to team
    team._ontology = ontology
    
    # Method to get concept insights for team
    def get_concept_insights(self) -> List[str]:
        """Get insights about concepts used by this team."""
        insights = []
        for concept in self.memory.ethical_concepts:
            info = ontology.get_concept_info(concept)
            if info:
                insights.append(
                    f"{concept.title()}: {info['definition']}\n"
                    f"  Related: {', '.join(info['related_concepts'])}"
                )
        return insights
    
    # Bind method to team
    team.get_concept_insights = lambda: get_concept_insights(team)
    
    return team
