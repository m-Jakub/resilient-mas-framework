"""
Team module for multi-agent debate system.
Implements shared memory and team-level coordination for philosopher agents.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from langchain.schema import HumanMessage


@dataclass
class TeamMemory:
    """
    Shared memory structure for a team.
    Stores team-level insights, arguments, and strategic information.
    """
    team_id: str
    insights: List[str] = field(default_factory=list)
    key_arguments: List[str] = field(default_factory=list)
    ethical_concepts: List[str] = field(default_factory=list)
    strategic_notes: Dict[str, Any] = field(default_factory=dict)
    deliberation_history: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: Optional[datetime] = None
    
    def add_insight(self, insight: str, author: str):
        """Add an insight to team memory."""
        self.insights.append(insight)
        self.deliberation_history.append({
            "type": "insight",
            "content": insight,
            "author": author,
            "timestamp": datetime.now()
        })
        self.last_updated = datetime.now()
    
    def add_argument(self, argument: str, author: str):
        """Add a key argument to team memory."""
        self.key_arguments.append(argument)
        self.deliberation_history.append({
            "type": "argument",
            "content": argument,
            "author": author,
            "timestamp": datetime.now()
        })
        self.last_updated = datetime.now()
    
    def add_ethical_concept(self, concept: str):
        """Track ethical concepts discussed by the team."""
        if concept not in self.ethical_concepts:
            self.ethical_concepts.append(concept)
            self.last_updated = datetime.now()
    
    def set_strategic_note(self, key: str, value: Any):
        """Set a strategic note (e.g., 'preferred_approach', 'weak_points')."""
        self.strategic_notes[key] = value
        self.last_updated = datetime.now()
    
    def get_summary(self) -> str:
        """Get a text summary of team memory for prompts."""
        summary = f"=== Team {self.team_id} Shared Memory ===\n\n"
        
        if self.insights:
            summary += "Key Insights:\n"
            for i, insight in enumerate(self.insights, 1):
                summary += f"{i}. {insight}\n"
            summary += "\n"
        
        if self.key_arguments:
            summary += "Team Arguments:\n"
            for i, arg in enumerate(self.key_arguments, 1):
                summary += f"{i}. {arg}\n"
            summary += "\n"
        
        if self.ethical_concepts:
            summary += f"Ethical Concepts in Focus: {', '.join(self.ethical_concepts)}\n\n"
        
        if self.strategic_notes:
            summary += "Strategic Notes:\n"
            for key, value in self.strategic_notes.items():
                summary += f"- {key}: {value}\n"
            summary += "\n"
        
        return summary
    
    def clear(self):
        """Clear team memory (for new debate rounds)."""
        self.insights.clear()
        self.key_arguments.clear()
        self.ethical_concepts.clear()
        self.strategic_notes.clear()
        self.deliberation_history.clear()
        self.last_updated = None


class Team:
    """
    Represents a team of philosopher agents with shared memory and coordination.
    """
    
    def __init__(self, team_id: str, name: str):
        """
        Initialize a team.
        
        Args:
            team_id: Unique identifier for the team
            name: Human-readable name for the team
        """
        self.team_id = team_id
        self.name = name
        self.agents: Dict[str, Any] = {}  # key -> PhilosopherAgent
        self.memory = TeamMemory(team_id=team_id)
        self.leader: Optional[str] = None  # Optional: agent key who speaks for team
    
    def add_agent(self, key: str, agent: Any):
        """
        Add an agent to the team.
        
        Args:
            key: Unique key for the agent (e.g., 'mill')
            agent: PhilosopherAgent instance
        """
        self.agents[key] = agent
        print(f"[Team {self.name}] Added agent: {agent.name}")
    
    def remove_agent(self, key: str) -> bool:
        """
        Remove an agent from the team.
        
        Args:
            key: Agent key to remove
            
        Returns:
            True if agent was removed, False if not found
        """
        if key in self.agents:
            agent_name = self.agents[key].name
            del self.agents[key]
            print(f"[Team {self.name}] Removed agent: {agent_name}")
            return True
        return False
    
    def set_leader(self, key: str):
        """
        Set a team leader (agent who represents team in debate).
        
        Args:
            key: Agent key to set as leader
        """
        if key in self.agents:
            self.leader = key
            print(f"[Team {self.name}] Leader set to: {self.agents[key].name}")
        else:
            raise ValueError(f"Agent '{key}' not found in team")
    
    def get_agent_keys(self) -> List[str]:
        """Get list of agent keys in the team."""
        return list(self.agents.keys())
    
    def get_agents(self) -> Dict[str, Any]:
        """Get dictionary of all agents."""
        return self.agents
    
    def get_memory_summary(self) -> str:
        """Get formatted summary of team's shared memory."""
        return self.memory.get_summary()
    
    def update_memory_from_deliberation(self, agent_key: str, content: str, content_type: str = "insight"):
        """
        Update team memory based on internal deliberation.
        
        Args:
            agent_key: Key of agent contributing
            content: Content to add to memory
            content_type: Type of content ('insight', 'argument', or 'concept')
        """
        agent_name = self.agents[agent_key].name if agent_key in self.agents else agent_key
        
        if content_type == "insight":
            self.memory.add_insight(content, agent_name)
        elif content_type == "argument":
            self.memory.add_argument(content, agent_name)
        elif content_type == "concept":
            self.memory.add_ethical_concept(content)
    
    def internal_deliberation(self, topic: str, rounds: int = 1) -> str:
        """
        Conduct internal team deliberation (narada).
        Agents exchange ideas and update shared memory.
        
        Args:
            topic: Topic/dilemma to deliberate on
            rounds: Number of deliberation rounds
            
        Returns:
            Summary of deliberation
        """
        print(f"\n{'='*60}")
        print(f"[Team {self.name}] Internal Deliberation Starting")
        print(f"Topic: {topic}")
        print(f"{'='*60}\n")
        
        deliberation_history = []
        
        for round_num in range(rounds):
            print(f"\n--- Deliberation Round {round_num + 1} ---\n")
            
            for agent_key, agent in self.agents.items():
                # Create deliberation prompt with team context
                from langchain.schema import HumanMessage
                
                deliberation_prompt = f"""
You are in an internal team deliberation (narada) with your teammates.

TEAM: {self.name}
TOPIC: {topic}

{self.memory.get_summary()}

PREVIOUS DELIBERATION:
{self._format_deliberation_history(deliberation_history)}

Your task: Share your initial thoughts, insights, or strategic considerations with your team.
Focus on building a strong collective understanding and strategy.
Keep your response concise (2-3 sentences).
"""
                
                # Agent responds to deliberation prompt
                # Create chat history with the deliberation prompt as first message
                chat_history = [HumanMessage(content=deliberation_prompt, name="Team")]
                
                response = agent.respond(
                    topic=topic,
                    chat_history=chat_history,
                    ontology_insights=[]
                )
                
                print(f"[{agent.name}]: {response}\n")
                
                # Store in deliberation history
                deliberation_history.append({
                    "agent": agent_key,
                    "name": agent.name,
                    "response": response
                })
                
                # Update team memory (extract key points)
                self.memory.add_insight(response[:200], agent.name)  # Store first 200 chars as insight
        
        print(f"\n{'='*60}")
        print(f"[Team {self.name}] Deliberation Complete")
        print(f"{'='*60}\n")
        
        # Store deliberation history for logging
        self._last_deliberation = deliberation_history
        
        return self._format_deliberation_summary(deliberation_history)
    
    def _format_deliberation_history(self, history: List[Dict[str, str]]) -> str:
        """Format deliberation history for prompts."""
        if not history:
            return "No previous deliberation yet."
        
        formatted = []
        for entry in history:
            formatted.append(f"[{entry['name']}]: {entry['response']}")
        return "\n".join(formatted)
    
    def _format_deliberation_summary(self, history: List[Dict[str, str]]) -> str:
        """Create a summary of the deliberation."""
        summary = f"Team {self.name} Deliberation Summary:\n"
        summary += f"Total contributions: {len(history)}\n"
        summary += f"Participants: {', '.join(set(e['name'] for e in history))}\n"
        return summary
    
    def __repr__(self):
        return f"Team(id='{self.team_id}', name='{self.name}', agents={len(self.agents)})"
