"""
Synchronized Speech Queue System for Multi-Agent Debates

This module implements a fair turn-taking system with speech queues for each team,
timeout handling, message limits, and moderator-controlled scheduling.

Architecture:
    - Each team has its own FIFO speech queue
    - Moderator acts as scheduler, allocating turns fairly
    - Timeout mechanism prevents hung turns
    - Message limits enforce brevity
    - Queue state tracked for logging and analysis

Turn Allocation Strategies:
    - Round-robin: Alternate between teams
    - Fair: Balance based on total speaking time
    - Priority: Allow moderator to prioritize certain speakers
    - AI-driven: LLM-based selection (existing system)
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import time


@dataclass
class QueuedSpeaker:
    """Represents an agent waiting to speak."""
    team_id: str
    agent_id: str
    priority: int = 0  # Higher = more urgent
    queued_at: datetime = field(default_factory=datetime.now)
    
    def wait_time_seconds(self) -> float:
        """Calculate how long this speaker has been waiting."""
        return (datetime.now() - self.queued_at).total_seconds()


@dataclass
class TurnAllocation:
    """Result of moderator's turn allocation decision."""
    team_id: str
    agent_id: str
    allocation_strategy: str  # "round_robin", "fair", "priority", "ai_driven"
    max_response_length: int = 1000  # Character limit for this turn
    timeout_seconds: int = 60  # Max time for LLM response
    metadata: Dict = field(default_factory=dict)


class SpeechQueueManager:
    """
    Manages synchronized speech queues for multi-team debates.
    
    Features:
    - FIFO queues per team
    - Fair turn allocation
    - Timeout detection
    - Message length limits
    - Queue state tracking
    - Multiple scheduling strategies
    
    Usage:
        manager = SpeechQueueManager(teams={"team_a": team_a, "team_b": team_b})
        manager.enqueue("team_a", "util_agent")
        allocation = manager.allocate_next_turn(strategy="round_robin")
        manager.mark_turn_complete(allocation.team_id, allocation.agent_id)
    """
    
    def __init__(self,
                 teams: Dict[str, any],  # team_id -> Team object
                 default_timeout_seconds: int = 60,
                 default_max_message_length: int = 1000,
                 enable_priority: bool = False):
        """
        Initialize speech queue manager.
        
        Args:
            teams: Dictionary of team_id -> Team object
            default_timeout_seconds: Default max time per turn
            default_max_message_length: Default character limit per message
            enable_priority: Whether to allow priority-based scheduling
        """
        self.teams = teams
        self.default_timeout = default_timeout_seconds
        self.default_max_length = default_max_message_length
        self.enable_priority = enable_priority
        
        # Speech queues: team_id -> deque of QueuedSpeaker
        self.queues: Dict[str, deque] = {
            team_id: deque() for team_id in teams.keys()
        }
        
        # Statistics
        self.turn_history: List[TurnAllocation] = []
        self.team_speaking_time: Dict[str, float] = {team_id: 0.0 for team_id in teams.keys()}
        self.team_turn_count: Dict[str, int] = {team_id: 0 for team_id in teams.keys()}
        
        # State tracking
        self.current_speaker: Optional[TurnAllocation] = None
        self.turn_start_time: Optional[datetime] = None
        self.last_team_to_speak: Optional[str] = None
    
    def enqueue(self, team_id: str, agent_id: str, priority: int = 0):
        """Add an agent to their team's speech queue."""
        if team_id not in self.queues:
            raise ValueError(f"Unknown team: {team_id}")
        
        speaker = QueuedSpeaker(
            team_id=team_id,
            agent_id=agent_id,
            priority=priority
        )
        self.queues[team_id].append(speaker)
    
    def enqueue_all_team_members(self):
        """Enqueue all agents from all teams (initialization)."""
        for team_id, team in self.teams.items():
            for agent_id in team.get_agent_keys():
                self.enqueue(team_id, agent_id)
    
    def dequeue(self, team_id: str) -> Optional[QueuedSpeaker]:
        """Remove and return the next speaker from a team's queue."""
        if team_id not in self.queues or not self.queues[team_id]:
            return None
        return self.queues[team_id].popleft()
    
    def peek(self, team_id: str) -> Optional[QueuedSpeaker]:
        """View the next speaker without removing them."""
        if team_id not in self.queues or not self.queues[team_id]:
            return None
        return self.queues[team_id][0]
    
    def get_queue_state(self) -> Dict[str, List[str]]:
        """Get current state of all queues."""
        return {
            team_id: [speaker.agent_id for speaker in queue]
            for team_id, queue in self.queues.items()
        }
    
    def get_queue_lengths(self) -> Dict[str, int]:
        """Get length of each team's queue."""
        return {team_id: len(queue) for team_id, queue in self.queues.items()}
    
    def is_queue_empty(self, team_id: str) -> bool:
        """Check if a team's queue is empty."""
        return len(self.queues.get(team_id, [])) == 0
    
    def are_all_queues_empty(self) -> bool:
        """Check if all queues are empty."""
        return all(len(queue) == 0 for queue in self.queues.values())
    
    def allocate_next_turn(self, strategy: str = "round_robin") -> Optional[TurnAllocation]:
        """
        Allocate next turn using specified strategy.
        
        Args:
            strategy: "round_robin", "fair", "priority", or "ai_driven"
        
        Returns:
            TurnAllocation with speaker and constraints, or None if no speakers available
        """
        if self.are_all_queues_empty():
            return None
        
        # Select strategy
        if strategy == "round_robin":
            return self._allocate_round_robin()
        elif strategy == "fair":
            return self._allocate_fair()
        elif strategy == "priority":
            return self._allocate_priority()
        else:
            # Default to round-robin
            return self._allocate_round_robin()
    
    def _allocate_round_robin(self) -> Optional[TurnAllocation]:
        """Alternate between teams."""
        # Find next team that has speakers
        team_ids = list(self.teams.keys())
        
        # Start from team after the last speaker
        if self.last_team_to_speak:
            start_idx = (team_ids.index(self.last_team_to_speak) + 1) % len(team_ids)
        else:
            start_idx = 0
        
        # Try each team in order
        for i in range(len(team_ids)):
            team_id = team_ids[(start_idx + i) % len(team_ids)]
            if not self.is_queue_empty(team_id):
                speaker = self.dequeue(team_id)
                return TurnAllocation(
                    team_id=speaker.team_id,
                    agent_id=speaker.agent_id,
                    allocation_strategy="round_robin",
                    max_response_length=self.default_max_length,
                    timeout_seconds=self.default_timeout
                )
        
        return None
    
    def _allocate_fair(self) -> Optional[TurnAllocation]:
        """Allocate to team with least speaking time."""
        # Find team with minimum speaking time that has speakers
        min_time = float('inf')
        selected_team = None
        
        for team_id in self.teams.keys():
            if not self.is_queue_empty(team_id):
                if self.team_speaking_time[team_id] < min_time:
                    min_time = self.team_speaking_time[team_id]
                    selected_team = team_id
        
        if selected_team:
            speaker = self.dequeue(selected_team)
            return TurnAllocation(
                team_id=speaker.team_id,
                agent_id=speaker.agent_id,
                allocation_strategy="fair",
                max_response_length=self.default_max_length,
                timeout_seconds=self.default_timeout,
                metadata={"team_speaking_time": self.team_speaking_time[selected_team]}
            )
        
        return None
    
    def _allocate_priority(self) -> Optional[TurnAllocation]:
        """Allocate to highest priority speaker across all teams."""
        if not self.enable_priority:
            return self._allocate_round_robin()
        
        max_priority = -1
        selected_team = None
        selected_speaker = None
        
        # Find highest priority speaker
        for team_id, queue in self.queues.items():
            if queue:
                speaker = queue[0]  # Peek
                if speaker.priority > max_priority:
                    max_priority = speaker.priority
                    selected_team = team_id
                    selected_speaker = speaker
        
        if selected_speaker:
            self.dequeue(selected_team)  # Remove from queue
            return TurnAllocation(
                team_id=selected_speaker.team_id,
                agent_id=selected_speaker.agent_id,
                allocation_strategy="priority",
                max_response_length=self.default_max_length,
                timeout_seconds=self.default_timeout,
                metadata={"priority": max_priority}
            )
        
        return None
    
    def start_turn(self, allocation: TurnAllocation):
        """Mark turn as started (for timeout tracking)."""
        self.current_speaker = allocation
        self.turn_start_time = datetime.now()
        self.last_team_to_speak = allocation.team_id
    
    def mark_turn_complete(self, team_id: str, agent_id: str, duration_seconds: float):
        """Mark turn as complete and update statistics. Re-enqueues speaker for cycling."""
        if self.current_speaker and self.current_speaker.team_id == team_id:
            self.turn_history.append(self.current_speaker)
            self.team_speaking_time[team_id] += duration_seconds
            self.team_turn_count[team_id] += 1
            self.current_speaker = None
            self.turn_start_time = None
            
            # Re-enqueue speaker at end of their team's queue (enables cycling)
            self.enqueue(team_id, agent_id, priority=0)
    
    def check_timeout(self) -> bool:
        """Check if current turn has exceeded timeout."""
        if not self.current_speaker or not self.turn_start_time:
            return False
        
        elapsed = (datetime.now() - self.turn_start_time).total_seconds()
        return elapsed > self.current_speaker.timeout_seconds
    
    def get_statistics(self) -> Dict:
        """Get queue and turn statistics."""
        return {
            "total_turns": len(self.turn_history),
            "team_turn_counts": self.team_turn_count.copy(),
            "team_speaking_times": self.team_speaking_time.copy(),
            "current_queue_lengths": self.get_queue_lengths(),
            "turns_by_strategy": self._count_turns_by_strategy()
        }
    
    def _count_turns_by_strategy(self) -> Dict[str, int]:
        """Count turns by allocation strategy."""
        counts = {}
        for turn in self.turn_history:
            strategy = turn.allocation_strategy
            counts[strategy] = counts.get(strategy, 0) + 1
        return counts
    
    def reset_queues(self):
        """Clear all queues (for new debate round)."""
        for queue in self.queues.values():
            queue.clear()
    
    def get_longest_waiting_speaker(self) -> Optional[Tuple[str, str, float]]:
        """Find the speaker who has been waiting longest."""
        longest_wait = 0.0
        longest_team = None
        longest_agent = None
        
        for team_id, queue in self.queues.items():
            if queue:
                speaker = queue[0]
                wait = speaker.wait_time_seconds()
                if wait > longest_wait:
                    longest_wait = wait
                    longest_team = team_id
                    longest_agent = speaker.agent_id
        
        if longest_team:
            return (longest_team, longest_agent, longest_wait)
        return None
