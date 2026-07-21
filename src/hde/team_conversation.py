"""
Team-based philosophical conversation using LangGraph.
Implements deliberation → debate flow with team coordination.

Flow Architecture:
    Team Deliberation → Socratic Moderation → Debate Phase
    
The Socratic moderator intervenes between deliberation and debate to:
- Deepen philosophical inquiry
- Clarify vague concepts
- Expose hidden assumptions
- Test consistency of arguments

Enhanced Turn-Taking:
- Synchronized speech queues (FIFO per team)
- Fair scheduling with multiple strategies (round-robin, fair, priority)
- Timeout handling for LLM calls
- Message length limits
- Structured conversation logging (JSON/CSV)
"""

from langchain.schema import AIMessage, HumanMessage
from src.common.philosopher_agents import create_agent
from src.hde.team import Team
from src.common.ontology import ETHICAL_CONCEPTS
from src.common.api_abstraction import get_philosopher_api
from config import LLM_MODEL, TEMPERATURE
import time
import random
from typing import List, Dict, TypedDict, Optional, Any
from langgraph.graph import StateGraph, END
from src.hde.socrates_moderator import SocraticModerator, integrate_moderator_with_team
from src.hde.speech_queue import SpeechQueueManager, TurnAllocation
from src.hde.conversation_logger import ConversationLogger
from src.kg_cfr.system_dispatcher import SystemDispatcher
from src.kg_cfr.aegis_drau_context import AegisDrauContextBuilder
from src.common.llm_retry import invoke_with_retry
from datetime import datetime
import os
from pathlib import Path


class TeamConversationState(TypedDict):
    """Extended state for team-based conversations."""
    teams: Dict[str, Team]  # team_id -> Team
    chat_history: List[HumanMessage | AIMessage]
    current_speaker_team: str  # Which team is speaking
    current_speaker_agent: str  # Which agent within team is speaking
    current_phase: str  # "deliberation", "moderation", or "debate"
    turn_count: int
    deliberation_count: int
    ontology_insights: List[str]
    topic: str
    moderation_questions: List[Dict[str, str]]  # Questions asked by moderator
    belief_models: Dict[str, Dict[str, str]]  # ToM-Lite: {agent_name: {self_stance, opponent_X, ...}}
    use_id_rag: bool  # ABLATION FLAG: Enable/disable ID-RAG for testing
    use_adaptive_idrag: bool  # ADAPTIVE FLAG: Enable adaptive vs static ID-RAG retrieval
    perturbation: Optional[Dict[str, any]]  # Legacy SysAR: kept for backwards compatibility
    shock_event: Optional[Dict[str, Any]]  # SystemDispatcher: S1-S3 shock payload
    module_labels: Dict[str, str]  # AEGIS/DRAU module labels for teams
    private_thought_buffer: Dict[str, Any]  # CFR: Private strategic insights per agent
    cfr_mode: str  # CFR: Ablation mode ('none', 'prompt-only', 'knowledge-grounded')
    cf_flags: Dict[str, Any]  # CFR: Metadata about CFR activations per turn


class TeamPhilosophicalConversation:
    """
    Manages team-based philosophical debates with deliberation → moderation → debate flow.
    
    Flow Architecture:
    1. DELIBERATION PHASE: Each team conducts internal deliberation
    2. MODERATION PHASE: Socratic moderator intervenes with reflective questions
    3. DEBATE PHASE: Teams engage in structured inter-team debate
    
    The Socratic moderator analyzes team deliberations and intervenes when:
    - Conceptual conflicts or contradictions are detected
    - Discussion lacks sufficient depth
    - Epistemic gaps appear (unsupported claims)
    - Teams over-rely on single concepts without exploration
    """
    
    def __init__(self, 
                 deliberation_rounds: int = 1,
                 debate_turns: int = 6,
                 enable_moderation: bool = True,
                 moderation_tone: str = "reflective",
                 enable_logging: bool = True,
                 session_id: Optional[str] = None,
                 turn_allocation_strategy: str = "round_robin",
                 max_message_length: int = 1000,
                 turn_timeout_seconds: int = 60,
                 use_tom: bool = True,
                 enable_adversarial_perturbations: bool = False,
                 perturbation_type: Optional[str] = None,
                 shock_seed: Optional[int] = None,
                 cfr_mode: str = "none"):
        """
        Initialize team conversation with enhanced turn-taking and logging.
        
        Args:
            deliberation_rounds: Number of deliberation rounds per team
            debate_turns: Total number of debate turns
            enable_moderation: Whether to enable Socratic moderator
            moderation_tone: Moderator tone ("reflective", "inquisitive", "challenging")
            enable_logging: Whether to enable structured logging
            session_id: Unique ID for this session (auto-generated if None)
            turn_allocation_strategy: "round_robin", "fair", or "priority"
            max_message_length: Max characters per agent response
            turn_timeout_seconds: Max seconds per turn (LLM timeout)
            use_tom: Enable Theory of Mind (ToM-Lite) for strategic opponent modeling
            enable_adversarial_perturbations: Enable SystemDispatcher shock injection
            perturbation_type: Shock type ("S1", "S2", "S3", or "auto")
            shock_seed: Random seed for reproducible shock sampling
        """
        self.deliberation_rounds = deliberation_rounds
        self.debate_turns = debate_turns
        self.enable_moderation = enable_moderation
        self.enable_logging = enable_logging
        self.turn_allocation_strategy = turn_allocation_strategy
        self.max_message_length = max_message_length
        self.turn_timeout_seconds = turn_timeout_seconds
        self.use_tom = use_tom  # Store ToM flag
        self.enable_adversarial_perturbations = enable_adversarial_perturbations
        self.perturbation_type = perturbation_type
        self.perturbation_injected = False  # Track if perturbation was injected
        self.cfr_mode = cfr_mode  # CFR ablation mode
        
        self.graph = self._build_graph()
        
        # Initialize LLMs using API abstraction (must precede CFR engine init)
        api = get_philosopher_api()
        self.moderator_llm = api.get_llm()
        self.analyzer_llm = api.get_llm()

        # Initialize Counterfactual Reflection (CFR) engine
        # Pass the LLM so Steps A and C use real generation, not templates.
        if self.cfr_mode != "none":
            from src.kg_cfr.counterfactual_reflection import CounterfactualReflectionEngine
            self.cr_engine = CounterfactualReflectionEngine(llm=self.moderator_llm)
        else:
            self.cr_engine = None

        # Initialize SystemDispatcher for shocks (S1-S3)
        self.system_dispatcher = SystemDispatcher(seed=shock_seed)
        self.context_builder = AegisDrauContextBuilder()
        
        # Initialize Socratic moderator
        self.socratic_moderator = SocraticModerator(
            tone_profile=moderation_tone
        ) if enable_moderation else None
        
        # Initialize speech queue manager (will be set with teams in start_debate)
        self.speech_queue: Optional[SpeechQueueManager] = None
        
        # Initialize conversation logger
        self.logger: Optional[ConversationLogger] = None
        if enable_logging:
            self.logger = ConversationLogger(
                session_id=session_id or f"debate_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                metadata={
                    "deliberation_rounds": deliberation_rounds,
                    "debate_turns": debate_turns,
                    "moderation_enabled": enable_moderation,
                    "turn_allocation_strategy": turn_allocation_strategy,
                    "tom_enabled": use_tom,  # Log ToM status in metadata
                    "cfr_mode": self.cfr_mode
                }
            )
    
    def _build_graph(self):
        """
        Build LangGraph state machine for team debates.
        
        Graph Structure:
            team_deliberation → [moderation_round] → debate_turn → 
            analyze_ontology → router → select_next_speaker → debate_turn
        """
        workflow = StateGraph(TeamConversationState)
        
        # Add nodes
        workflow.add_node("team_deliberation", self._team_deliberation_phase)
        workflow.add_node("moderation_round", self._moderation_round)
        workflow.add_node("counterfactual_reflection", self._counterfactual_reflection)
        workflow.add_node("debate_turn", self._debate_turn)
        workflow.add_node("analyze_ontology", self._analyze_with_ontology)
        workflow.add_node("select_next_speaker", self._select_next_speaker)
        
        # Entry point: deliberation
        workflow.set_entry_point("team_deliberation")
        
        # Flow: deliberation -> moderation (if enabled) -> CFR -> debate turn
        if self.enable_moderation:
            workflow.add_edge("team_deliberation", "moderation_round")
            workflow.add_edge("moderation_round", "counterfactual_reflection")
        else:
            workflow.add_edge("team_deliberation", "counterfactual_reflection")
        
        workflow.add_edge("counterfactual_reflection", "debate_turn")
        
        # debate turn -> ontology analysis
        workflow.add_edge("debate_turn", "analyze_ontology")
        
        # After analysis: continue or end
        workflow.add_conditional_edges(
            "analyze_ontology",
            self._router,
            {
                "continue": "select_next_speaker",
                "end": END
            }
        )
        
        # After speaker selection, run CFR before the next debate turn
        workflow.add_edge("select_next_speaker", "counterfactual_reflection")
        
        # Compile (recursion_limit set in invoke config)
        return workflow.compile()
    
    def _team_deliberation_phase(self, state: TeamConversationState) -> TeamConversationState:
        """
        Phase 1: All teams conduct internal deliberation.
        Each team builds shared understanding before debate.
        """
        print("\n" + "="*70)
        print("PHASE 1: TEAM DELIBERATION")
        print("="*70)
        
        topic = state["topic"]
        deliberation_turn = 0  # Counter for deliberation messages
        
        for team_id, team in state["teams"].items():
            print(f"\n{'─'*70}")
            print(f"Team: {team.name}")
            print(f"{'─'*70}")
            
            # Conduct internal deliberation
            team.internal_deliberation(topic, rounds=self.deliberation_rounds)
            
            # Log deliberation turns
            if self.logger and hasattr(team, '_last_deliberation'):
                for entry in team._last_deliberation:
                    agent_key = entry['agent']
                    agent_name = entry['name']
                    response = entry['response']
                    
                    self.logger.log_turn(
                        turn_number=deliberation_turn,
                        phase="deliberation",
                        team_id=team_id,
                        team_name=team.name,
                        agent_id=agent_key,
                        agent_name=agent_name,
                        message_content=response,
                        response_time_ms=0,  # Not tracked for deliberation
                        concepts_mentioned=[]
                    )
                    deliberation_turn += 1
            
            # Small delay for rate limiting
            time.sleep(2)
        
        print("\n" + "="*70)
        print("DELIBERATION PHASE COMPLETE")
        print("="*70)
        
        # Transition to moderation or debate phase
        if self.enable_moderation:
            state["current_phase"] = "moderation"
        else:
            state["current_phase"] = "debate"
        
        state["deliberation_count"] = len(state["teams"])
        
        # Select first speaker (random team, random agent)
        first_team = random.choice(list(state["teams"].keys()))
        first_agent = random.choice(state["teams"][first_team].get_agent_keys())
        state["current_speaker_team"] = first_team
        state["current_speaker_agent"] = first_agent
        
        return state
    
    def _moderation_round(self, state: TeamConversationState) -> TeamConversationState:
        """
        Phase 1.5: Socratic Moderator intervenes with reflective questions.
        
        The moderator analyzes each team's deliberation and poses Socratic
        questions to deepen inquiry before the debate begins.
        
        Intervention Triggers:
        - Conceptual conflicts or contradictions
        - Insufficient conceptual depth
        - Epistemic gaps (unsupported claims)
        - Over-reliance on single concepts
        """
        print("\n" + "="*70)
        print("PHASE 1.5: SOCRATIC MODERATION")
        print("="*70)
        print("The moderator reviews team deliberations and poses reflective questions.")
        print("="*70)
        
        if not self.socratic_moderator:
            # Skip if moderation disabled
            state["current_phase"] = "debate"
            return state
        
        topic = state["topic"]
        moderation_questions = state.get("moderation_questions", [])
        
        # Intervene in each team's deliberation
        for team_id, team in state["teams"].items():
            print(f"\n{'─'*70}")
            print(f"Analyzing Team: {team.name}")
            print(f"{'─'*70}")
            
            # Check if intervention is needed (now returns dict with RAG metadata)
            intervention = self.socratic_moderator.intervene_in_discussion(
                team=team,
                topic=topic,
                phase="post-deliberation"
            )
            
            if intervention:
                question_type = intervention["question_type"]
                question = intervention["question"]
                rag_meta = intervention.get("rag_metadata", {})
                
                print(f"\n[MODERATOR - {question_type.upper()}]")
                print(f"  {question}")
                if rag_meta.get("rag_docs_count", 0) > 0:
                    print(f"  [RAG: {rag_meta['rag_docs_count']} docs retrieved]")
                print()
                
                # Log the question
                moderation_questions.append({
                    "team_id": team_id,
                    "team_name": team.name,
                    "type": question_type,
                    "question": question,
                    "rag_grounded": rag_meta.get("rag_docs_count", 0) > 0
                })
                
                # Log to conversation logger with RAG metadata
                if self.logger:
                    self.logger.log_moderation(
                        team_id=team_id,
                        team_name=team.name,
                        intervention_type=question_type,
                        question=question,
                        team_response=f"Team {team.name} reflecting on {question_type} question",
                        rag_grounded=rag_meta.get("rag_docs_count", 0) > 0,
                        rag_query=rag_meta.get("rag_query", ""),
                        retrieved_sources=rag_meta.get("rag_sources", []),
                        metadata={
                            "phase": "post-deliberation",
                            "rag_docs_count": rag_meta.get("rag_docs_count", 0),
                            "intervention_score": rag_meta.get("intervention_score", 0.0)
                        }
                    )
                
                # Integrate question into team memory
                integrate_moderator_with_team(team, self.socratic_moderator, question)
                
                # Simulate team reflection (optional: could invoke agents here)
                print(f"[{team.name}] Reflecting on moderator's question...")
                
                # Record a placeholder response
                reflection = f"Team {team.name} is considering the moderator's {question_type} question about their position."
                self.socratic_moderator.record_team_response(team_id, reflection)
            else:
                print(f"[MODERATOR] No intervention needed—{team.name}'s deliberation is sufficiently deep.")
        
        # Update state
        state["moderation_questions"] = moderation_questions
        state["current_phase"] = "debate"
        
        print("\n" + "="*70)
        print("MODERATION ROUND COMPLETE")
        print("="*70)
        
        # Show moderation summary
        if moderation_questions:
            print(f"\nTotal moderator interventions: {len(moderation_questions)}")
            for mq in moderation_questions:
                print(f"  • [{mq['team_name']}] {mq['type']}: {mq['question'][:80]}...")
        
        return state
    
    def _counterfactual_reflection(self, state: TeamConversationState) -> TeamConversationState:
        """
        CFR Node: Generate private strategic insights for current speaker.
        This runs BEFORE debate_turn and writes to private_thought_buffer.
        """
        cfr_mode = state.get("cfr_mode", "none")
        if cfr_mode == "none" or not self.cr_engine:
            return state
        
        team_id = state["current_speaker_team"]
        agent_key = state["current_speaker_agent"]
        
        if not team_id or not agent_key:
            return state
        
        last_msg = state["chat_history"][-1].content if state["chat_history"] else ""
        shock_event = state.get("shock_event") or state.get("perturbation")
        perturbation_active = bool(shock_event)

        attack_vector = self.context_builder.build_attack_vector(state, team_id)
        last_msg = attack_vector or last_msg
        
        # Build KG-CFR retriever (scenario-driven) if needed
        rag_retriever = None
        if cfr_mode == "knowledge-grounded":
            team = state.get("teams", {}).get(team_id)
            agent = team.agents.get(agent_key) if team else None
            if agent and getattr(agent, "identity_graph", None):
                def rag_retriever(query: str) -> List[str]:
                    query_lc = (query or "").lower()
                    keywords = [k for k in ETHICAL_CONCEPTS.keys() if k.lower() in query_lc]
                    search_terms = keywords if keywords else query_lc.split()[:5]
                    evidence_nodes: List[Dict[str, Any]] = []
                    for term in search_terms:
                        evidence_nodes.extend(agent.identity_graph.get_beliefs_about_concept(term))
                    evidence_texts = [n.get("content", "") for n in evidence_nodes if n.get("content")]
                    return evidence_texts[:8]
            else:
                print(f"[CFR] KG-CFR disabled for {team_id}:{agent_key} (missing identity_graph)")

        # Run CFR engine
        cr_result = self.cr_engine.run(
            topic=state["topic"],
            last_message=last_msg,
            turn_idx=state["turn_count"],
            debate_turns=self.debate_turns,
            belief_models=state.get("belief_models"),
            ontology_insights=state.get("ontology_insights"),
            cfr_mode=cfr_mode,
            rag_retriever=rag_retriever,
            perturbation_active=perturbation_active
        )
        
        if cr_result:
            # Store in private buffer (agent-specific key)
            agent_full_name = f"{team_id}:{agent_key}"
            state.setdefault("private_thought_buffer", {})[agent_full_name] = {
                "private_strategy": cr_result.strategic_insight,
                "flags": cr_result.guardrail_flags,
                "audit": cr_result.audit,
                "stable": cr_result.result.get("verdict_stable", True),
                "leak": cr_result.guardrail_flags.get("leak_risk", False)
            }
            
            # Store turn-level flags for logging
            state["cf_flags"] = {
                "activated": True,
                "turn_idx": state["turn_count"],
                "cfr_mode": cfr_mode,
                **cr_result.guardrail_flags,
                "stable": cr_result.result.get("verdict_stable", True),
                "leak_risk": cr_result.guardrail_flags.get("leak_risk", False),
                "kg_cfr": cr_result.audit.get("kg_cfr", False),
                "rag_docs_count": cr_result.audit.get("rag_docs_count", 0)
            }
            
            print(f"\n[CFR] Activated for {agent_full_name} (trigger: {cr_result.audit['trigger_reason']})")
        else:
            skip_reason = "no_result"
            if cfr_mode == "knowledge-grounded" and rag_retriever is None:
                skip_reason = "missing_retriever"
            state["cf_flags"] = {
                "activated": False,
                "turn_idx": state["turn_count"],
                "cfr_mode": cfr_mode,
                "skip_reason": skip_reason
            }
        
        return state
    
    def _debate_turn(self, state: TeamConversationState) -> TeamConversationState:
        """
        Phase 2: Agent from current team takes debate turn with timeout and logging.
        Uses team's shared memory as context.
        """
        # ========== SystemDispatcher: Inject S1-S3 Shock ==========
        # Inject shock after Turn 3 (before Turn 4)
        current_turn = state["turn_count"]
        
        if (self.enable_adversarial_perturbations and 
            not self.perturbation_injected and 
            current_turn == 4 and
            self.perturbation_type):
            
            shock_event = self.system_dispatcher.sample_shock(
                turn_number=current_turn,
                shock_type=self.perturbation_type
            )

            # Store shock in state (and legacy perturbation field for compatibility)
            state["shock_event"] = {
                "shock_id": shock_event.shock_id,
                "type": shock_event.type,
                "text": shock_event.text,
                "metadata": shock_event.metadata,
                "seed": shock_event.seed,
                "turn": shock_event.turn,
                "ts": shock_event.ts,
            }
            state["perturbation"] = state["shock_event"]

            # Add shock to chat history
            shock_msg = HumanMessage(
                content=shock_event.text,
                name="SystemDispatcher_Shock"
            )
            state["chat_history"].append(shock_msg)
            
            # Mark as injected
            self.perturbation_injected = True
            
            # Log perturbation if logging enabled
            if self.logger:
                self.logger.log_perturbation(
                    turn_number=current_turn,
                    perturbation_type=shock_event.type,
                    perturbation_text=shock_event.text
                )
        
        team_id = state["current_speaker_team"]
        agent_key = state["current_speaker_agent"]
        
        team = state["teams"][team_id]
        agent = team.agents[agent_key]
        
        print(f"\n{'─'*70}")
        print(f"[{team.name}] {agent.name}")
        print(f"{'─'*70}")
        
        # Track turn timing
        turn_start = datetime.now()
        
        # Build context with module memory
        team_memory_context = team.get_memory_summary()
        
        # Create enhanced chat history with module context
        chat_history = state["chat_history"].copy()
        
        # Add module memory as system context for this turn
        if team_memory_context:
            memory_msg = HumanMessage(
                content=f"[MODULE CONTEXT]\n{team_memory_context}",
                name="ModuleMemory"
            )
            # Insert module context before the last message
            if len(chat_history) > 1:
                chat_history.insert(-1, memory_msg)
            else:
                chat_history.append(memory_msg)

        module_perspective = self.context_builder.build_module_perspective(state, team_id)
        if module_perspective:
            module_msg = HumanMessage(
                content=f"[AEGIS MODULE]\n{module_perspective}",
                name="AegisModule"
            )
            chat_history.append(module_msg)
        
        # Add length constraint instruction (soft limit for LLM)
        # Estimate: ~4 chars per token, so max_message_length chars ≈ max_message_length/4 tokens
        estimated_token_limit = self.max_message_length // 4
        length_instruction = HumanMessage(
            content=f"[INSTRUCTION: Keep your response under {estimated_token_limit} tokens (~{self.max_message_length} characters). Be concise but complete your argument.]",
            name="System"
        )
        chat_history.append(length_instruction)
        
        # ========== CFR: Extract Private Strategy ==========
        private_strategy = ""
        agent_full_name = f"{team_id}:{agent_key}"
        buf = state.get("private_thought_buffer", {})
        if agent_full_name in buf and isinstance(buf[agent_full_name], dict):
            private_strategy = buf[agent_full_name].get("private_strategy", "")[:300]
        
        # ========== ToM-Lite Integration ==========
        # Check if ToM is enabled and belief_models are available
        use_tom_active = (self.use_tom and 
                         "belief_models" in state and 
                         len(state["belief_models"]) > 0)
        
        if use_tom_active:
            # Get current agent's belief model
            agent_full_name = f"{team_id}:{agent_key}"
            belief_model = state["belief_models"].get(agent_full_name, {})
            
            if belief_model:
                agent_stance = belief_model.get("self_stance", "unknown")
                
                # Find last opponent message from chat history
                opponent_name = None
                opponent_argument = None
                opponent_stance = "unknown"
                
                # Iterate backwards to find last message from opposing team
                for msg in reversed(state["chat_history"]):
                    if hasattr(msg, 'name') and msg.name and ':' in msg.name:
                        msg_team_id, msg_agent_key = msg.name.split(':')
                        if msg_team_id != team_id:  # Found opponent
                            opponent_name = msg.name
                            opponent_argument = msg.content[:300]  # First 300 chars
                            opponent_stance = belief_model.get(f"opponent_{opponent_name}", "unknown")
                            break
                
                # Build ToM strategic prompt if we found opponent
                if opponent_name and opponent_argument:
                    tom_strategic_prompt = self._build_tom_strategy_prompt(
                        agent_name=agent.name,
                        agent_stance=agent_stance,
                        opponent_name=opponent_name,
                        opponent_stance=opponent_stance,
                        opponent_last_argument=opponent_argument
                    )
                    
                    # Inject ToM strategy into chat history
                    tom_msg = HumanMessage(
                        content=tom_strategic_prompt,
                        name="ToM_Strategy"
                    )
                    chat_history.append(tom_msg)
                    
                    print(f"[ToM-Lite] {agent.name} ({agent_stance}) vs {opponent_name} ({opponent_stance})")
        
        # Agent responds with full context (with timeout handling)
        try:
            # Rate limiting: add small delay to avoid hitting API limits (15 req/min)
            # With 4 agents + moderator, we can hit ~5-7 calls per turn
            if os.getenv("DISABLE_RATE_LIMIT_WAIT", "0").lower() not in {"1", "true", "yes"}:
                time.sleep(1.5)  # Ensures we stay under 15/min (1 req every ~4 seconds)
            
            response = agent.respond(
                topic=state["topic"],
                chat_history=chat_history,
                ontology_insights=state.get("ontology_insights", []),
                use_id_rag=state.get("use_id_rag", True),  # ABLATION FLAG propagation
                use_adaptive_idrag=state.get("use_adaptive_idrag", True),  # ADAPTIVE FLAG propagation
                private_strategy=private_strategy  # CFR: Private strategic insight
            )
            
            # ========== CRITICAL: Block turn progression on API failures ==========
            # Check if agent returned an API error message (rate limit or retry failure)
            if "paused due to API rate limit" in response or "failed after" in response and "retries" in response:
                error_type = "rate limit" if "paused" in response else "retry failure"
                print(f"\n{'='*70}")
                print(f"❌ CRITICAL ERROR: {agent.name} {error_type}")
                print(f"{'='*70}")
                print(f"Response: {response}")
                print(f"\n⚠️ DEBATE HALTED - Cannot proceed without complete turn")
                print(f"{'='*70}\n")
                
                # Mark conversation as failed
                state["conversation_ended"] = True
                state["error"] = f"{agent.name} API failure: {error_type}"
                
                # Log error if logging enabled
                if self.logger:
                    self.logger.log_error(
                        turn_number=state["turn_count"],
                        agent_name=agent.name,
                        error_type=error_type,
                        error_message=response
                    )
                
                return state  # Return immediately without updating history or incrementing turn
            
            # Enforce message length limit with smart truncation (end at sentence)
            if len(response) > self.max_message_length:
                original_length = len(response)
                # Try to find last sentence boundary before limit
                truncated = response[:self.max_message_length]
                
                # Find last sentence-ending punctuation
                last_period = truncated.rfind('.')
                last_exclaim = truncated.rfind('!')
                last_question = truncated.rfind('?')
                
                # Use the latest sentence boundary found
                sentence_end = max(last_period, last_exclaim, last_question)
                
                if sentence_end > self.max_message_length * 0.7:  # Keep if at least 70% of limit
                    response = truncated[:sentence_end + 1]  # Include the punctuation
                else:
                    response = truncated + "..."
                
                print(f"[WARNING] Response truncated from {original_length} to {len(response)} chars")
            
        except Exception as e:
            print(f"[ERROR] Agent response failed: {e}")
            response = f"[Agent {agent.name} encountered an error during response]"
        
        # Calculate response time
        turn_end = datetime.now()
        response_time_ms = int((turn_end - turn_start).total_seconds() * 1000)
        
        print(response)
        print(f"{'─'*70}")
        print(f"[Response time: {response_time_ms}ms | Length: {len(response)} chars]")
        print()
        
        # Log the turn, including Gemini usage_metadata if available
        if self.logger:
            concepts = team.memory.ethical_concepts[-5:] if team.memory.ethical_concepts else []
            cf_flags = state.get("cf_flags", {}) if isinstance(state.get("cf_flags", {}), dict) else {}
            turn_metadata = {
                "cf_flags": cf_flags
            }
            # Try to extract usage_metadata from agent (set by LLM invocation)
            usage_meta = getattr(agent, "last_usage_metadata", None)
            if usage_meta:
                turn_metadata["usage_metadata"] = usage_meta

            self.logger.log_turn(
                turn_number=state["turn_count"],
                phase="debate",
                team_id=team_id,
                team_name=team.name,
                agent_id=agent_key,
                agent_name=agent.name,
                message_content=str(response),
                concepts_mentioned=concepts,
                response_time_ms=response_time_ms,
                metadata=turn_metadata
            )

            if cf_flags.get("activated"):
                verdict_stable = bool(cf_flags.get("stable", True))
                leak_risk = bool(cf_flags.get("leak_risk", False))
                agent_full_name = f"{team_id}:{agent_key}"
                self.logger.log_counterfactual_turn(
                    turn_number=state["turn_count"],
                    agent_id=agent_full_name,
                    flags=cf_flags,
                    verdict_stable=verdict_stable,
                    leak_risk=leak_risk
                )
        
        # Update speech queue statistics
        if self.speech_queue:
            duration = (turn_end - turn_start).total_seconds()
            self.speech_queue.mark_turn_complete(team_id, agent_key, duration)
            
            # Log queue state
            if self.logger:
                self.logger.log_queue_state(
                    turn_number=state["turn_count"],
                    team_queues=self.speech_queue.get_queue_state(),
                    current_speaker={"team_id": team_id, "agent_id": agent_key}
                )
        
        # Update state
        state["chat_history"].append(
            AIMessage(
                content=response,
                name=f"{team_id}:{agent_key}"
            )
        )
        state["turn_count"] += 1
        state["ontology_insights"] = []
        
        # ========== CFR: Clean private buffer after turn ==========
        if agent_full_name in state.get("private_thought_buffer", {}):
            state["private_thought_buffer"].pop(agent_full_name, None)
        
        return state
    
    def _analyze_with_ontology(self, state: TeamConversationState) -> TeamConversationState:
        """Analyze last message for ethical concepts."""
        if not state["chat_history"]:
            return state
        
        last_msg = state["chat_history"][-1].content
        concept_keys = list(ETHICAL_CONCEPTS.keys())
        
        prompt = f"""You are a philosophical concept analyzer.
Identify key ethical concepts from the text.
Respond with a comma-separated list of concept keys ONLY.

Available Concepts: {', '.join(concept_keys)}

Text:
---
{last_msg}
---

Concept keys:"""
        
        try:
            resp = invoke_with_retry(
                self.analyzer_llm.invoke,
                prompt,
                operation="ontology.analyzer_llm.invoke",
            )
            found = [k.strip() for k in resp.content.split(',') if k.strip() in concept_keys]
            insights = []
            
            if found:
                print(f"\n[Ontology] Identified: {', '.join(found)}")
                for key in found:
                    info = ETHICAL_CONCEPTS[key]
                    insights.append(
                        f"Concept: {key.title()}\n"
                        f"Definition: {info['definition']}\n"
                        f"Primary School: {info['primary_school'].title()}"
                    )
                    
                    # Update team memory with identified concepts
                    team_id = state["current_speaker_team"]
                    state["teams"][team_id].memory.add_ethical_concept(key)
            
            state["ontology_insights"] = insights
        except Exception as e:
            print(f"[Ontology] Error: {e}")
            state["ontology_insights"] = []
        
        return state
    
    def _router(self, state: TeamConversationState) -> str:
        """
        Decide whether to continue debate or end.
        Ends when: turn limit reached OR all speech queues are empty.
        """
        # Check turn limit
        if state["turn_count"] >= self.debate_turns:
            print(f"\n[Router] Turn limit reached ({state['turn_count']}/{self.debate_turns})")
            return "end"
        
        # Check if all queues are empty (no more speakers available)
        if self.speech_queue:
            queue_lengths = self.speech_queue.get_queue_lengths()
            total_queued = sum(queue_lengths.values())
            
            if total_queued == 0:
                print(f"\n[Router] All speech queues empty - ending debate early")
                print(f"[Router] Completed {state['turn_count']} turns (limit was {self.debate_turns})")
                return "end"
        
        return "continue"
    
    # ========== ToM-Lite Helper Functions ==========
    
    def _extract_stance_from_identity(self, agent) -> str:
        """
        Extract philosophical stance label from agent's identity graph.
        Uses heuristic keyword matching on core beliefs.
        
        Returns:
            Stance label: "deontological", "utilitarian", "virtue_ethics", 
                         "natural_law", "will_to_power", "unknown"
        """
        if not hasattr(agent, 'identity_graph') or not agent.identity_graph:
            return "unknown"
        
        # PhilosopherIdentityGraph is an object, not a dict
        # Use get_core_beliefs() method to retrieve core belief nodes
        core_beliefs_text = []
        
        try:
            core_beliefs = agent.identity_graph.get_core_beliefs()
            for belief in core_beliefs:
                # Each belief is an IdentityNode with 'content' attribute
                if hasattr(belief, 'content'):
                    core_beliefs_text.append(belief.content.lower())
        except Exception as e:
            print(f"[WARN] Failed to extract core beliefs from identity graph: {e}")
            return "unknown"
        
        combined_text = ' '.join(core_beliefs_text)
        
        if not combined_text:
            return "unknown"
        
        # Heuristic keyword matching (order matters - check most specific first)
        if any(kw in combined_text for kw in ['categorical imperative', 'duty', 'maxim', 'moral law']):
            return "deontological"
        elif any(kw in combined_text for kw in ['utility', 'happiness', 'greatest number', 'pleasure', 'consequence']):
            return "utilitarian"
        elif any(kw in combined_text for kw in ['virtue', 'eudaimonia', 'character', 'golden mean', 'flourishing']):
            return "virtue_ethics"
        elif any(kw in combined_text for kw in ['divine', 'natural law', 'eternal law', 'god', 'grace']):
            return "natural_law"
        elif any(kw in combined_text for kw in ['will to power', 'übermensch', 'master morality', 'slave morality']):
            return "will_to_power"
        else:
            return "unknown"
    
    def _build_tom_strategy_prompt(self, 
                                    agent_name: str,
                                    agent_stance: str,
                                    opponent_name: str,
                                    opponent_stance: str,
                                    opponent_last_argument: str) -> str:
        """
        Build strategic ToM prompt for generating counter-argument strategy.
        
        This prompt implements ToM-Lite: explicit belief model of opponent
        used to generate targeted counter-arguments.
        
        Args:
            agent_name: Current agent's name
            agent_stance: Current agent's philosophical stance (from belief_models)
            opponent_name: Opposing agent's name
            opponent_stance: Opposing agent's stance (from belief_models)
            opponent_last_argument: Last argument made by opponent
        
        Returns:
            Strategic prompt string with ToM context
        """
        # HARDENING: Get core terminology for anti-mimicry guardrails
        agent_terms = self._get_core_terms(agent_stance)
        opponent_terms = self._get_core_terms(opponent_stance)
        
        tom_prompt = f"""[ToM STRATEGIC CONTEXT]
You are {agent_name}. Your philosophical stance is {agent_stance}.

Your opponent {opponent_name} (stance: {opponent_stance}) just argued:
"{opponent_last_argument}"

STRATEGIC OBJECTIVE: Generate a counter-argument strategy that exploits the weaknesses of {opponent_stance}.

Known weaknesses of {opponent_stance}:
"""
        
        # Add stance-specific weakness knowledge
        weakness_map = {
            "deontological": "- Rigid rules may conflict with context-specific moral intuitions\n- Cannot handle moral dilemmas where duties conflict\n- Ignores consequences even when catastrophic",
            "utilitarian": "- Permits sacrificing individual rights for collective benefit\n- Difficult to calculate all consequences accurately\n- May justify morally repugnant acts if they maximize utility",
            "virtue_ethics": "- Vague guidance on specific actions\n- Relies on subjective character judgments\n- Cultural relativism about virtues",
            "natural_law": "- Appeals to divine authority may not convince non-believers\n- Assumes teleological worldview not universally accepted\n- Rigid natural order may conflict with human autonomy",
            "will_to_power": "- Justifies exploitation of the weak\n- Lacks universal moral principles\n- Nihilistic implications undermine social cooperation"
        }
        
        tom_prompt += weakness_map.get(opponent_stance, "- General philosophical inconsistencies\n- Empirical claims without evidence\n- Logical fallacies")
        
        # HARDENING: Add anti-mimicry guardrails
        tom_prompt += f"""

YOUR STRATEGY: Craft your response to directly challenge these weaknesses in {opponent_name}'s {opponent_stance} position while defending your {agent_stance} approach.

⚠️ IDENTITY GUARDRAILS (CRITICAL):
1. Counter-argue FROM YOUR {agent_stance.upper()} FRAMEWORK ONLY
2. Use YOUR core terminology: {', '.join(agent_terms)}
3. NEVER adopt their terminology ({', '.join(opponent_terms)}) as your own
4. If critiquing their concepts, use quotes: "their notion of '{opponent_terms[0]}'"
5. If mimicry detected in your draft, REWRITE using your doctrinal vocabulary

EXAMPLE (if you are Deontological critiquing Utilitarian):
❌ BAD: "Utility demands we respect dignity" (mimicry - using 'utility')
✅ GOOD: "Duty demands we respect dignity, regardless of their utilitarian calculus"
"""
        
        return tom_prompt
    
    def _get_core_terms(self, stance: str) -> list:
        """
        Get core terminology for a philosophical stance.
        
        HARDENING: Enables anti-mimicry guardrails in ToM prompts.
        
        Args:
            stance: Philosophical stance (e.g., 'deontological', 'utilitarian')
            
        Returns:
            List of 3-5 core terms for that stance
        """
        terms_map = {
            "deontological": ["duty", "categorical imperative", "dignity", "rational law"],
            "utilitarian": ["utility", "happiness", "greatest good", "consequences"],
            "virtue_ethics": ["virtue", "character", "flourishing", "excellence", "eudaimonia"],
            "natural_law": ["natural law", "divine order", "eternal law", "teleology"],
            "will_to_power": ["will to power", "master morality", "übermensch", "strength"]
        }
        
        return terms_map.get(stance, ["core principles", "framework", "doctrine"])
    
    def _select_next_speaker(self, state: TeamConversationState) -> TeamConversationState:
        """Select next speaker using speech queue manager."""
        print("\n[Moderator] Selecting next speaker...")
        
        # Use speech queue if available
        if self.speech_queue:
            allocation = self.speech_queue.allocate_next_turn(
                strategy=self.turn_allocation_strategy
            )
            
            if allocation:
                state["current_speaker_team"] = allocation.team_id
                state["current_speaker_agent"] = allocation.agent_id
                
                team = state["teams"][allocation.team_id]
                agent = team.agents[allocation.agent_id]
                
                print(f"[Moderator] Selected ({allocation.allocation_strategy}): {team.name} - {agent.name}")
                print(f"  Max length: {allocation.max_response_length} chars | Timeout: {allocation.timeout_seconds}s")
                
                # Start turn timing
                self.speech_queue.start_turn(allocation)
                
                # Show queue status
                queue_lengths = self.speech_queue.get_queue_lengths()
                print(f"  Queue status: {queue_lengths}")
                
                return state
            else:
                print("[Moderator] All queues empty!")
                return state
        
        # Fallback: Original AI moderator selection (if queue not initialized)
        print("[Moderator] Using AI-based selection (queue not initialized)...")
        
        # Build team and agent options
        team_options = []
        for team_id, team in state["teams"].items():
            for agent_key in team.get_agent_keys():
                agent = team.agents[agent_key]
                team_options.append({
                    "team_id": team_id,
                    "team_name": team.name,
                    "agent_key": agent_key,
                    "agent_name": agent.name
                })
        
        # Format for prompt
        options_text = "\n".join(
            f"- {opt['team_id']}:{opt['agent_key']} ({opt['team_name']} - {opt['agent_name']})"
            for opt in team_options
        )
        
        # Recent history (last 3 turns)
        recent_history = state["chat_history"][-3:] if len(state["chat_history"]) > 3 else state["chat_history"]
        history_text = "\n".join(
            f"{msg.name}: {msg.content[:150]}..."
            for msg in recent_history
            if hasattr(msg, 'name')
        )
        
        prompt = f"""You are a debate moderator. Select the next speaker for balanced discussion.

Participants:
{options_text}

Recent exchanges:
---
{history_text}
---

Respond with ONE option in format: team_id:agent_key
Example: team_a:mill"""
        
        try:
            time.sleep(2)  # Rate limiting
            resp = invoke_with_retry(
                self.moderator_llm.invoke,
                prompt,
                operation="moderator_llm.invoke",
            )
            selection = resp.content.strip().lower()
            
            # Parse selection
            if ':' in selection:
                team_id, agent_key = selection.split(':', 1)
                if team_id in state["teams"] and agent_key in state["teams"][team_id].agents:
                    state["current_speaker_team"] = team_id
                    state["current_speaker_agent"] = agent_key
                    team = state["teams"][team_id]
                    agent = team.agents[agent_key]
                    print(f"[Moderator] Selected: {team.name} - {agent.name}")
                    return state
        except Exception as e:
            print(f"[Moderator] Error: {e}")
        
        # Fallback: random selection
        print("[Moderator] Fallback: random selection")
        team_id = random.choice(list(state["teams"].keys()))
        agent_key = random.choice(state["teams"][team_id].get_agent_keys())
        state["current_speaker_team"] = team_id
        state["current_speaker_agent"] = agent_key
        
        return state
    
    def start_debate(self,
                     topic: str,
                     teams: Dict[str, Team],
                     use_id_rag: bool = True,
                     use_adaptive_idrag: bool = True) -> Dict:
        """
        Start a team-based debate with speech queue and logging.
        
        Args:
            topic: The ethical dilemma to debate
            teams: Dictionary of team_id -> Team
            use_id_rag: Enable ID-RAG identity context (default True, False for ablation study)
            use_adaptive_idrag: Enable adaptive vs static ID-RAG (default True, False for baseline)
            
        Returns:
            Final state with chat history and team memories
        """
        # Initialize speech queue manager
        self.speech_queue = SpeechQueueManager(
            teams=teams,
            default_timeout_seconds=self.turn_timeout_seconds,
            default_max_message_length=self.max_message_length
        )
        self.speech_queue.enqueue_all_team_members()
        
        # Update logger with topic
        if self.logger:
            self.logger.topic = topic
        
        initial_state = TeamConversationState(
            teams=teams,
            chat_history=[HumanMessage(content=topic, name="Moderator")],
            current_speaker_team="",
            current_speaker_agent="",
            current_phase="deliberation",
            turn_count=0,
            deliberation_count=0,
            ontology_insights=[],
            topic=topic,
            moderation_questions=[],
            belief_models={},  # Initialize empty belief models
            use_id_rag=use_id_rag,  # ABLATION FLAG
            use_adaptive_idrag=use_adaptive_idrag,  # ADAPTIVE FLAG
            perturbation=None,  # Legacy SysAR: populated with shock_event for compatibility
            shock_event=None,  # SystemDispatcher: S1-S3 shock payload
            module_labels=self.context_builder.build_module_labels(teams),
            private_thought_buffer={},  # CFR: Initialize empty private buffer
            cfr_mode=self.cfr_mode,  # CFR: Ablation mode
            cf_flags={}  # CFR: Initialize empty flags
        )
        
        # ========== Auto-Initialize ToM-Lite Belief Models ==========
        if self.use_tom:
            print("\n[ToM-Lite] Initializing belief models from identity graphs...")
            
            for team_id, team in teams.items():
                for agent_key, agent in team.agents.items():
                    agent_full_name = f"{team_id}:{agent_key}"
                    
                    # Extract self stance from identity graph
                    self_stance = self._extract_stance_from_identity(agent)
                    
                    # Initialize belief model for this agent
                    initial_state["belief_models"][agent_full_name] = {
                        "self_stance": self_stance
                    }
                    
                    print(f"  {agent.name} ({agent_full_name}): {self_stance}")
                    
                    # Extract stances for all opposing agents
                    for opponent_team_id, opponent_team in teams.items():
                        if opponent_team_id != team_id:  # Only opposing teams
                            for opponent_key, opponent_agent in opponent_team.agents.items():
                                opponent_full_name = f"{opponent_team_id}:{opponent_key}"
                                opponent_stance = self._extract_stance_from_identity(opponent_agent)
                                
                                # Store opponent model in current agent's belief_models
                                initial_state["belief_models"][agent_full_name][f"opponent_{opponent_full_name}"] = opponent_stance
                                
                                print(f"    vs {opponent_agent.name} ({opponent_full_name}): {opponent_stance}")
            
            print("[ToM-Lite] Belief models initialized [OK]\n")
        else:
            print("\n[ToM-Lite] DISABLED - Using baseline (no opponent modeling)\n")
        
        print("\n" + "="*70)
        print("TEAM-BASED PHILOSOPHICAL DEBATE")
        print("="*70)
        print(f"\nTopic: {topic}\n")
        print(f"Teams: {', '.join(t.name for t in teams.values())}")
        print(f"Deliberation rounds: {self.deliberation_rounds}")
        print(f"Debate turns: {self.debate_turns}")
        print(f"Turn allocation strategy: {self.turn_allocation_strategy}")
        print(f"Message limit: {self.max_message_length} chars")
        print(f"Timeout: {self.turn_timeout_seconds}s")
        print(f"ToM-Lite: {'enabled' if self.use_tom else 'disabled'}")
        print(f"Logging: {'enabled' if self.logger else 'disabled'}")
        print("="*70)
        
        # Execute graph with config (thread_id + recursion_limit)
        config = {
            "configurable": {"thread_id": self.logger.session_id if self.logger else "default"},
            "recursion_limit": 50  # Increased from default 25 for longer debates
        }
        final_state = self.graph.invoke(initial_state, config=config)
        
        print("\n" + "="*70)
        print("DEBATE COMPLETE")
        print("="*70)
        
        # Print moderation summary if enabled
        if self.enable_moderation and self.socratic_moderator:
            print("\n" + "="*70)
            print("SOCRATIC MODERATION SUMMARY")
            print("="*70)
            print(self.socratic_moderator.summarize_moderation_round())
        
        # Print speech queue statistics
        if self.speech_queue:
            print("\n" + "="*70)
            print("SPEECH QUEUE STATISTICS")
            print("="*70)
            stats = self.speech_queue.get_statistics()
            print(f"Total turns: {stats['total_turns']}")
            print(f"Turn counts by team: {stats['team_turn_counts']}")
            print(f"Speaking time by team: {stats['team_speaking_times']}")
            print(f"Turns by strategy: {stats['turns_by_strategy']}")
        
        # Save logs if logging enabled
        if self.logger:
            print("\n" + "="*70)
            print("SAVING CONVERSATION LOGS")
            print("="*70)
            
            # Determine appropriate subdirectory based on session_id
            logs_base = Path("logs")
            if self.logger.session_id.startswith("test_"):
                logs_dir = logs_base / "test_sessions"
            elif self.logger.session_id.startswith("exp_"):
                logs_dir = logs_base / "experiments"
            elif self.logger.session_id.startswith("debate_"):
                logs_dir = logs_base / "production"
            else:
                logs_dir = logs_base / "test_sessions"  # Default
            
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Save JSON
            json_path = logs_dir / f"{self.logger.session_id}.json"
            self.logger.save_json(json_path)
            
            # Save CSV
            self.logger.save_csv(logs_dir)
            
            # Print summary
            self.logger.print_summary()
        
        # Print team memories
        print("\n" + "="*70)
        print("FINAL TEAM MEMORIES")
        print("="*70)
        for team_id, team in final_state["teams"].items():
            print(f"\n{team.name}:")
            print(team.get_memory_summary())
        
        return final_state


def run_league_of_epochs_debate(dilemma: str,
                                 team_configs: List[Dict] = None,
                                 deliberation_rounds: int = 1,
                                 debate_turns: int = 6,
                                 enable_moderation: bool = True,
                                 enable_logging: bool = True,
                                 session_id: Optional[str] = None,
                                 use_tom: bool = True,
                                 cfr_mode: str = "none",
                                 shock_seed: Optional[int] = None,
                                 use_id_rag: bool = True,
                                 use_adaptive_idrag: bool = True) -> Dict:
    """
    Main entry point for running League of Epochs debate (Mode B - Experimental Condition).
    
    This function orchestrates the full team-based debate with Socratic moderation
    for A/B testing against the baseline tutor.
    
    Args:
        dilemma: Ethical dilemma to debate
        team_configs: Team configurations (defaults to standard 2-team setup)
        deliberation_rounds: Number of internal deliberation rounds per team
        debate_turns: Total number of inter-team debate turns
        enable_moderation: Whether to enable Socratic moderator interventions
        enable_logging: Whether to save structured logs
        session_id: Unique session identifier (auto-generated if None)
        use_tom: Enable Theory of Mind (ToM-Lite) for strategic opponent modeling
    
    Returns:
        Dictionary containing final state, conversation log, and metadata
    """
    print("=" * 80)
    print("LEAGUE OF EPOCHS DEBATE MODE (Experimental Condition - Mode B)")
    print("=" * 80)
    print(f"Dilemma: {dilemma}")
    print(f"Deliberation rounds: {deliberation_rounds}")
    print(f"Debate turns: {debate_turns}")
    print(f"Moderation: {'enabled' if enable_moderation else 'disabled'}")
    print(f"ToM-Lite: {'enabled' if use_tom else 'disabled'}")
    print(f"CFR mode: {cfr_mode}")
    print("=" * 80)
    
    # Default team configuration if not provided
    # Balanced: 2 agents per team for fair comparison
    if team_configs is None:
        team_configs = [
            {
                "team_id": "team_consequentialist",
                "team_name": "Consequentialist Coalition",
                "agent_keys": ["mill", "st_augustine"]  # Mill + Augustine (virtue ethics angle)
            },
            {
                "team_id": "team_deontological",
                "team_name": "Deontological Alliance",
                "agent_keys": ["kant", "aquinas"]  # Kant + Aquinas
            }
        ]
    
    # Create teams
    from src.common.philosopher_agents import create_agent
    
    teams = {}
    for config in team_configs:
        team = Team(
            team_id=config["team_id"],
            name=config.get("team_name", config["team_id"])
        )
        
        # Add agents to team
        for agent_key in config["agent_keys"]:
            agent = create_agent(agent_key)
            if agent:
                team.add_agent(agent_key, agent)
        
        teams[config["team_id"]] = team
    
    # Initialize conversation
    conversation = TeamPhilosophicalConversation(
        deliberation_rounds=deliberation_rounds,
        debate_turns=debate_turns,
        enable_moderation=enable_moderation,
        enable_logging=enable_logging,
        session_id=session_id,
        use_tom=use_tom,  # Pass ToM flag to conversation
        cfr_mode=cfr_mode,
        shock_seed=shock_seed
    )
    
    # Run debate
    final_state = conversation.start_debate(
        dilemma,
        teams,
        use_id_rag=use_id_rag,
        use_adaptive_idrag=use_adaptive_idrag
    )
    
    # Extract conversation log for analysis
    conversation_log = []
    for i, msg in enumerate(final_state["chat_history"]):
        speaker = getattr(msg, 'name', 'Unknown')
        content = msg.content
        
        conversation_log.append({
            "turn": i + 1,
            "speaker": speaker,
            "message": content,
            "message_type": "HumanMessage" if isinstance(msg, HumanMessage) else "AIMessage"
        })
    
    # Compile metadata
    metadata = {
        "mode": "debate",
        "dilemma": dilemma,
        "teams": [
            {
                "team_id": tid,
                "team_name": team.name,
                "agents": [agent.name for agent in team.agents.values()],
                "final_memory": team.get_memory_summary()
            }
            for tid, team in final_state["teams"].items()
        ],
        "total_turns": final_state["turn_count"],
        "deliberation_rounds": deliberation_rounds,
        "debate_turns": debate_turns,
        "moderation_enabled": enable_moderation,
        "tom_enabled": use_tom,  # Log ToM status
        "cfr_mode": cfr_mode,
        "belief_models": final_state.get("belief_models", {}),  # Log final belief models
        "moderation_questions": final_state.get("moderation_questions", []),
        "ontology_insights": final_state.get("ontology_insights", []),
        "conversation_log": conversation_log,
        "session_id": conversation.logger.session_id if conversation.logger else session_id
    }
    
    print("\n" + "=" * 80)
    print(f"LEAGUE OF EPOCHS SESSION COMPLETE: {metadata['total_turns']} turns")
    print("=" * 80)
    
    return metadata


def create_team_debate(team_configs: List[Dict],
                       deliberation_rounds: int = 1,
                       debate_turns: int = 6) -> TeamPhilosophicalConversation:
    """
    Factory function to create team debate.
    
    Args:
        team_configs: List of team configurations, e.g.:
            [
                {
                    "team_id": "team_consequentialist",
                    "name": "Consequentialists",
                    "agents": ["mill"]
                },
                {
                    "team_id": "team_deontologist",
                    "name": "Deontologists",
                    "agents": ["kant"]
                }
            ]
        deliberation_rounds: Rounds of internal deliberation
        debate_turns: Total debate turns
    
    Returns:
        TeamPhilosophicalConversation instance with teams configured
    """
    conversation = TeamPhilosophicalConversation(
        deliberation_rounds=deliberation_rounds,
        debate_turns=debate_turns
    )
    
    return conversation

