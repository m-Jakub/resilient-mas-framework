"""
Structured Conversation Logger for Multi-Agent Debates

This module provides comprehensive logging capabilities for philosophical debates,
capturing all interactions, moderator interventions, and ontology analyses in
structured formats (JSON and CSV) for analysis and reproducibility.

Logging Schema:
    - Turn-level logs: speaker, message, timestamp, phase
    - Moderation logs: intervention type, question, team response
    - Ontology logs: concepts discussed, relationships detected
    - Metadata: timing, token counts, queue states

Output Formats:
    - JSON: Full nested structure with all metadata
    - CSV: Flattened for easy analysis in spreadsheets/pandas
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict, field


@dataclass
class TurnLog:
    """Record of a single debate turn."""
    turn_number: int
    timestamp: str
    phase: str  # "deliberation", "moderation", "debate"
    team_id: str
    team_name: str
    agent_id: str
    agent_name: str
    message_content: str
    message_length: int
    concepts_mentioned: List[str] = field(default_factory=list)
    response_time_ms: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModerationLog:
    """Record of a moderator intervention."""
    timestamp: str
    team_id: str
    team_name: str
    intervention_type: str  # "definitional", "dialectical", etc.
    question: str
    team_response: Optional[str] = None
    rag_grounded: bool = False
    rag_query: Optional[str] = None  # Query used for RAG retrieval
    retrieved_sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OntologyLog:
    """Record of ontology analysis at a point in time."""
    timestamp: str
    turn_number: int
    concepts_discussed: List[str]
    relationships_detected: List[Dict[str, str]]
    conflicts_identified: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueueStateLog:
    """Record of speech queue state at a point in time."""
    timestamp: str
    turn_number: int
    team_queues: Dict[str, List[str]]  # team_id -> list of agent_ids in queue
    current_speaker: Optional[Dict[str, str]] = None  # team_id, agent_id
    queue_lengths: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationLogger:
    """
    Comprehensive logger for multi-agent philosophical debates.
    
    Features:
    - Turn-by-turn logging of all speeches
    - Moderation intervention tracking
    - Ontology analysis logging
    - Speech queue state tracking
    - Export to JSON and CSV formats
    - Automatic timestamping and metadata capture
    
    Usage:
        logger = ConversationLogger(session_id="trolley_debate_001")
        logger.log_turn(turn_number=1, phase="deliberation", ...)
        logger.log_moderation(team_id="team_a", ...)
        logger.save_json("logs/debate_001.json")
        logger.save_csv("logs/debate_001_turns.csv")
    """
    
    def __init__(self, 
                 session_id: str,
                 topic: str = "",
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize conversation logger.
        
        Args:
            session_id: Unique identifier for this debate session
            topic: The ethical dilemma being debated
            metadata: Additional session-level metadata
        """
        self.session_id = session_id
        self.topic = topic
        self.session_metadata = metadata or {}
        self.session_start = datetime.now().isoformat()
        
        # Logs
        self.turn_logs: List[TurnLog] = []
        self.moderation_logs: List[ModerationLog] = []
        self.ontology_logs: List[OntologyLog] = []
        self.queue_state_logs: List[QueueStateLog] = []
        
        # Statistics
        self.stats = {
            "total_turns": 0,
            "total_moderator_interventions": 0,
            "total_concepts_discussed": 0,
            "session_duration_seconds": 0,
            "cf_activations": 0,
            "cc_count": 0,
            "cc_sum": 0.0,
            "cc_alpha": self.session_metadata.get("cc_alpha", 0.5)
        }

    def log_counterfactual_turn(
        self,
        turn_number: int,
        agent_id: str,
        flags: Dict[str, Any],
        verdict_stable: bool,
        leak_risk: bool,
        alpha: Optional[float] = None
    ) -> None:
        """
        Log Counterfactual Reflection (CFR) activation and update CC metric.

        Args:
            turn_number: Turn index
            agent_id: Agent key (team_id:agent_id)
            flags: Guardrail flags from CFR (identity_ok, leak_risk, doctrinal_drift)
            verdict_stable: Whether verdict remained stable under CFR
            leak_risk: Whether CFR flagged potential leak/drift
            alpha: Penalty coefficient for CC (defaults to session metadata or 0.5)
        """
        effective_alpha = alpha if alpha is not None else self.stats.get("cc_alpha", 0.5)
        stability_value = 1.0 if verdict_stable else 0.0
        leak_value = 1.0 if leak_risk else 0.0
        cc_component = stability_value - (effective_alpha * leak_value)

        self.stats["cf_activations"] += 1
        self.stats["cc_count"] += 1
        self.stats["cc_sum"] += cc_component

        # Store per-turn CFR info in session metadata for traceability
        self.session_metadata.setdefault("cr_turns", []).append({
            "turn_number": turn_number,
            "agent_id": agent_id,
            "flags": flags,
            "verdict_stable": verdict_stable,
            "leak_risk": leak_risk,
            "cc_component": cc_component
        })
    
    def log_turn(self,
                 turn_number: int,
                 phase: str,
                 team_id: str,
                 team_name: str,
                 agent_id: str,
                 agent_name: str,
                 message_content: str,
                 concepts_mentioned: Optional[List[str]] = None,
                 response_time_ms: Optional[int] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """Log a single debate turn."""
        turn_log = TurnLog(
            turn_number=turn_number,
            timestamp=datetime.now().isoformat(),
            phase=phase,
            team_id=team_id,
            team_name=team_name,
            agent_id=agent_id,
            agent_name=agent_name,
            message_content=message_content,
            message_length=len(message_content),
            concepts_mentioned=concepts_mentioned or [],
            response_time_ms=response_time_ms,
            metadata=metadata or {}
        )
        self.turn_logs.append(turn_log)
        self.stats["total_turns"] += 1

    def log_error(self,
                  turn_number: int,
                  agent_name: str,
                  error_type: str,
                  error_message: str) -> None:
        """Log a runtime error during a debate turn."""
        self.session_metadata.setdefault("errors", []).append({
            "timestamp": datetime.now().isoformat(),
            "turn_number": turn_number,
            "agent_name": agent_name,
            "error_type": error_type,
            "error_message": error_message
        })
    
    def log_perturbation(self,
                         turn_number: int,
                         perturbation_type: str,
                         perturbation_text: str):
        """
        Log an adversarial perturbation for SysAR testing.
        
        Args:
            turn_number: Turn when perturbation was injected
            perturbation_type: Type of perturbation
            perturbation_text: Full text of perturbation question
        """
        # Store as special metadata
        if "perturbation" not in self.stats:
            self.stats["perturbation"] = {
                "injected_at_turn": turn_number,
                "type": perturbation_type,
                "text": perturbation_text,
                "timestamp": datetime.now().isoformat()
            }
    
    def log_moderation(self,
                       team_id: str,
                       team_name: str,
                       intervention_type: str,
                       question: str,
                       team_response: Optional[str] = None,
                       rag_grounded: bool = False,
                       rag_query: Optional[str] = None,
                       retrieved_sources: Optional[List[str]] = None,
                       metadata: Optional[Dict[str, Any]] = None):
        """Log a moderator intervention."""
        mod_log = ModerationLog(
            timestamp=datetime.now().isoformat(),
            team_id=team_id,
            team_name=team_name,
            intervention_type=intervention_type,
            question=question,
            team_response=team_response,
            rag_grounded=rag_grounded,
            rag_query=rag_query,
            retrieved_sources=retrieved_sources or [],
            metadata=metadata or {}
        )
        self.moderation_logs.append(mod_log)
        self.stats["total_moderator_interventions"] += 1
    
    def log_ontology(self,
                     turn_number: int,
                     concepts_discussed: List[str],
                     relationships_detected: Optional[List[Dict[str, str]]] = None,
                     conflicts_identified: Optional[List[str]] = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """Log ontology analysis at a point in time."""
        onto_log = OntologyLog(
            timestamp=datetime.now().isoformat(),
            turn_number=turn_number,
            concepts_discussed=concepts_discussed,
            relationships_detected=relationships_detected or [],
            conflicts_identified=conflicts_identified or [],
            metadata=metadata or {}
        )
        self.ontology_logs.append(onto_log)
        self.stats["total_concepts_discussed"] = len(set(concepts_discussed))
    
    def log_queue_state(self,
                        turn_number: int,
                        team_queues: Dict[str, List[str]],
                        current_speaker: Optional[Dict[str, str]] = None,
                        metadata: Optional[Dict[str, Any]] = None):
        """Log speech queue state."""
        queue_log = QueueStateLog(
            timestamp=datetime.now().isoformat(),
            turn_number=turn_number,
            team_queues=team_queues,
            current_speaker=current_speaker,
            queue_lengths={team_id: len(queue) for team_id, queue in team_queues.items()},
            metadata=metadata or {}
        )
        self.queue_state_logs.append(queue_log)
    
    def finalize_session(self):
        """Finalize session and calculate duration."""
        session_end = datetime.now().isoformat()
        start_dt = datetime.fromisoformat(self.session_start)
        end_dt = datetime.fromisoformat(session_end)
        self.stats["session_duration_seconds"] = (end_dt - start_dt).total_seconds()
        if self.stats.get("cc_count", 0) > 0:
            self.stats["cc_score"] = self.stats["cc_sum"] / self.stats["cc_count"]
        else:
            self.stats["cc_score"] = None
        self.session_metadata["session_end"] = session_end
    
    def get_summary(self) -> Dict[str, Any]:
        """Get session summary with statistics."""
        stats = self._normalize_metric_keys(self.stats)
        return {
            "session_id": self.session_id,
            "topic": self.topic,
            "session_start": self.session_start,
            "statistics": stats,
            "metadata": self.session_metadata
        }

    def _normalize_metric_keys(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Rename metric keys for anonymization (SysAR -> PRR, DA -> ACA)."""
        normalized = dict(stats)
        key_map = {
            "SysAR": "Perturbation_Rebound_Rate",
            "sysar": "Perturbation_Rebound_Rate",
            "DA": "Axiomatic_Constraint_Adherence",
            "da": "Axiomatic_Constraint_Adherence",
        }

        for old_key, new_key in key_map.items():
            if old_key in normalized:
                normalized[new_key] = normalized.pop(old_key)

        return normalized
    
    def _determine_log_subdirectory(self) -> str:
        """Determine appropriate subdirectory based on session_id."""
        if self.session_id.startswith("test_"):
            return "test_sessions"
        elif self.session_id.startswith("exp_"):
            return "experiments"
        elif self.session_id.startswith("debate_"):
            return "production"
        else:
            # Default to test_sessions for unknown formats
            return "test_sessions"
    
    def save_json(self, output_path: Path):
        """Save complete session log to JSON."""
        self.finalize_session()
        
        output_data = {
            "session_info": self.get_summary(),
            "turn_logs": [asdict(log) for log in self.turn_logs],
            "moderation_logs": [asdict(log) for log in self.moderation_logs],
            "ontology_logs": [asdict(log) for log in self.ontology_logs],
            "queue_state_logs": [asdict(log) for log in self.queue_state_logs]
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved JSON log to: {output_path}")
    
    def save_csv(self, output_dir: Path):
        """Save logs to CSV files (separate files for turns, moderation, etc.)."""
        self.finalize_session()
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save turn logs
        turns_csv = output_dir / f"{self.session_id}_turns.csv"
        with open(turns_csv, 'w', newline='', encoding='utf-8') as f:
            if self.turn_logs:
                writer = csv.DictWriter(f, fieldnames=asdict(self.turn_logs[0]).keys())
                writer.writeheader()
                for log in self.turn_logs:
                    row = asdict(log)
                    # Flatten lists for CSV
                    row['concepts_mentioned'] = ', '.join(row['concepts_mentioned'])
                    row['metadata'] = json.dumps(row['metadata'])
                    writer.writerow(row)
        
        print(f"✓ Saved turns CSV to: {turns_csv}")
        
        # Save moderation logs
        mod_csv = output_dir / f"{self.session_id}_moderation.csv"
        with open(mod_csv, 'w', newline='', encoding='utf-8') as f:
            if self.moderation_logs:
                writer = csv.DictWriter(f, fieldnames=asdict(self.moderation_logs[0]).keys())
                writer.writeheader()
                for log in self.moderation_logs:
                    row = asdict(log)
                    row['retrieved_sources'] = ', '.join(row['retrieved_sources'])
                    row['metadata'] = json.dumps(row['metadata'])
                    writer.writerow(row)
        
        print(f"✓ Saved moderation CSV to: {mod_csv}")
        
        # Save ontology logs
        onto_csv = output_dir / f"{self.session_id}_ontology.csv"
        with open(onto_csv, 'w', newline='', encoding='utf-8') as f:
            if self.ontology_logs:
                writer = csv.DictWriter(f, fieldnames=asdict(self.ontology_logs[0]).keys())
                writer.writeheader()
                for log in self.ontology_logs:
                    row = asdict(log)
                    row['concepts_discussed'] = ', '.join(row['concepts_discussed'])
                    row['relationships_detected'] = json.dumps(row['relationships_detected'])
                    row['conflicts_identified'] = ', '.join(row['conflicts_identified'])
                    row['metadata'] = json.dumps(row['metadata'])
                    writer.writerow(row)
        
        print(f"✓ Saved ontology CSV to: {onto_csv}")
        
        # Save queue state logs
        queue_csv = output_dir / f"{self.session_id}_queue_states.csv"
        with open(queue_csv, 'w', newline='', encoding='utf-8') as f:
            if self.queue_state_logs:
                # Flatten queue state for CSV
                fieldnames = ['timestamp', 'turn_number', 'current_speaker_team', 
                             'current_speaker_agent', 'queue_lengths', 'metadata']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for log in self.queue_state_logs:
                    row = {
                        'timestamp': log.timestamp,
                        'turn_number': log.turn_number,
                        'current_speaker_team': log.current_speaker['team_id'] if log.current_speaker else '',
                        'current_speaker_agent': log.current_speaker['agent_id'] if log.current_speaker else '',
                        'queue_lengths': json.dumps(log.queue_lengths),
                        'metadata': json.dumps(log.metadata)
                    }
                    writer.writerow(row)
        
        print(f"✓ Saved queue states CSV to: {queue_csv}")
        
        # Save session summary
        summary_csv = output_dir / f"{self.session_id}_summary.csv"
        with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
            summary = self.get_summary()
            # Build fieldnames dynamically to include perturbation if present
            fieldnames = ['session_id', 'topic', 'session_start', 'total_turns', 
                         'total_moderator_interventions', 'total_concepts_discussed',
                         'session_duration_seconds', 'cf_activations', 'cc_score', 'cc_alpha']
            
            # Add perturbation fields if present
            if 'perturbation' in self.stats:
                fieldnames.extend(['perturbation_type', 'perturbation_injected_at_turn', 'perturbation_text'])
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Build row with only known fields from statistics
            stats = summary['statistics']
            row = {
                'session_id': summary['session_id'],
                'topic': summary['topic'],
                'session_start': summary['session_start'],
                'total_turns': stats.get('total_turns', 0),
                'total_moderator_interventions': stats.get('total_moderator_interventions', 0),
                'total_concepts_discussed': stats.get('total_concepts_discussed', 0),
                'session_duration_seconds': stats.get('session_duration_seconds', 0.0),
                'cf_activations': stats.get('cf_activations', 0),
                'cc_score': stats.get('cc_score', None),
                'cc_alpha': stats.get('cc_alpha', 0.5)
            }
            
            # Add perturbation data if present
            if 'perturbation' in self.stats:
                pert = self.stats['perturbation']
                row['perturbation_type'] = pert.get('type', '')
                row['perturbation_injected_at_turn'] = pert.get('injected_at_turn', '')
                row['perturbation_text'] = pert.get('text', '')[:100] + '...'  # Truncate long text
            
            writer.writerow(row)
        
        print(f"✓ Saved session summary to: {summary_csv}")
    
    def print_summary(self):
        """Print session summary to console."""
        summary = self.get_summary()
        print("\n" + "="*70)
        print("CONVERSATION LOG SUMMARY")
        print("="*70)
        print(f"Session ID: {summary['session_id']}")
        print(f"Topic: {summary['topic']}")
        print(f"Start: {summary['session_start']}")
        print(f"\nStatistics:")
        for key, value in summary['statistics'].items():
            print(f"  • {key}: {value}")
        print("="*70)
