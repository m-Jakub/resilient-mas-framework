from langchain.schema import AIMessage, HumanMessage
from src.common.philosopher_agents import create_agent
from src.common.ontology import ETHICAL_CONCEPTS
from langchain_google_genai import ChatGoogleGenerativeAI
from config import LLM_MODEL, TEMPERATURE
import time  # for rate-limit delay
import random  # needed for speaker selection and initial choice
from typing import List, TypedDict, Annotated
from langgraph.graph import StateGraph, END

# =============================================================================
# DEPRECATED: This module is legacy code for single-agent conversations.
# Use TeamPhilosophicalConversation from team_conversation.py instead.
#
# This class is maintained only for backward compatibility with --mode=legacy.
# New code should use the team-based debate system with:
# - Team + TeamMemory for episodic memory
# - SpeechQueueManager for turn-taking
# - SocraticModerator for RAG-grounded moderation
# - ID-RAG integration for long-term identity
# =============================================================================

# Define the state for the graph.
# This is the central object that will be passed between nodes.
class ConversationState(TypedDict):
    participants: dict
    chat_history: List[HumanMessage | AIMessage]
    current_speaker: str
    turn_count: int
    ontology_insights: List[str]

class PhilosophicalConversation:
    """
    DEPRECATED: Manages and orchestrates conversations using a LangGraph state machine.
    
    This is the legacy single-agent conversation system. For new implementations,
    use TeamPhilosophicalConversation from team_conversation.py which provides:
    - Team-based deliberation and debate
    - Socratic moderation with RAG
    - Fair turn-taking via SpeechQueueManager
    - ID-RAG integration for philosopher identity
    """
    def __init__(self, total_turns: int = 8):
        self.total_turns = total_turns
        self.graph = self._build_graph()
        self.moderator_llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.3)
        self.analyzer_llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.3) # another LLM for ontology analysis
        # For interactive participant selection
        self.initial_participants: List[str] = []

    def _build_graph(self):
        """Builds the LangGraph state machine."""
        workflow = StateGraph(ConversationState)

        # Add nodes: turn-taking, analysis, and speaker selection
        workflow.add_node("take_turn", self._agent_turn)
        workflow.add_node("analyze_with_ontology", self._analyze_with_ontology)
        workflow.add_node("select_speaker", self._select_next_speaker_node)
        
        # Entry point: agent turn
        workflow.set_entry_point("take_turn")

        # Flow: take turn -> ontology analysis
        workflow.add_edge("take_turn", "analyze_with_ontology")

        # After analysis, decide to continue or end
        workflow.add_conditional_edges(
            "analyze_with_ontology",
            self._router,  # only checks turn count
            {
                "continue": "select_speaker",
                "end": END
            }
        )
        # After selecting speaker, next turn
        workflow.add_edge("select_speaker", "take_turn")
        
        return workflow.compile()

    # New analysis node: extract ontology insights after each turn
    def _analyze_with_ontology(self, state: ConversationState) -> ConversationState:
        """Analyzes the last message and adds ontology insights."""
        last_msg = state["chat_history"][-1].content
        concept_keys = list(ETHICAL_CONCEPTS.keys())
        prompt = f"""You are a philosophical concept analyzer.
        Your task is to identify key ethical concepts from the provided text.
        Read the following text and identify which of the following concepts are being discussed.
        Respond with a comma-separated list of the concepts you identify.
        Do NOT explain your reasoning. Only return the concept keys.

Available Concepts: {', '.join(concept_keys)}

Text to Analyze:
---
{last_msg}
---

Respond with a comma-separated list of concept keys."""
        try:
            resp = self.analyzer_llm.invoke(prompt)
            found = [k.strip() for k in resp.content.split(',') if k.strip() in concept_keys]
            insights = []
            if found:
                print(f"Ontology Analyzer found: {', '.join(found)}")
                for key in found:
                    info = ETHICAL_CONCEPTS[key]
                    insights.append(
                        f"Concept: {key.title()}\nDefinition: {info['definition']}\nPrimary School: {info['primary_school'].title()}"
                    )
            state["ontology_insights"] = insights
        except Exception as e:
            print(f"Error during ontology analysis: {e}")
            state["ontology_insights"] = []
        return state

    def _agent_turn(self, state: ConversationState) -> ConversationState:
        """A single agent takes its turn."""
        speaker_key = state["current_speaker"]
        participant = state["participants"][speaker_key]
        
        print(f"\n--- Speaker: {participant.name} ---")
        print("-" * 40)
        
        # Pass the ontology insights into the agent's respond method
        response = participant.respond(
            topic="",  # Topic is in the history
            chat_history=state["chat_history"],
            ontology_insights=state.get("ontology_insights", [])
        )
        print(response)
        
        # Update the history and turn count
        state["chat_history"].append(AIMessage(content=response, name=speaker_key))
        state["turn_count"] += 1
        
        # Clear insights after they've been used
        state["ontology_insights"] = []
        return state

    def _router(self, state: ConversationState) -> str:
        """Router now only decides to continue or end based on turn count."""
        return "end" if state["turn_count"] >= self.total_turns else "continue"

    # Removed old speaker selection; using _select_next_speaker_node instead
    
    def _select_next_speaker_node(self, state: ConversationState) -> ConversationState:
        """Selects the next speaker and updates the state."""
        print("\n(Moderator is thinking...)")
        time.sleep(15) # time delay for rate-limiting stability
        participants = state["participants"]
        chat_history = state["chat_history"]
        participant_options = ", ".join(participants.keys())
        participant_descriptions = "\n".join(
            f"- {key}: {agent.name}" for key, agent in participants.items()
        )
        formatted_history = "\n".join(
            f"{(msg.name if hasattr(msg, 'name') and msg.name else 'Human')}: {msg.content}"
            for msg in chat_history
        )
        prompt = f"""You are an expert debate moderator. Select the next speaker for a balanced conversation.

Here are the participants:
{participant_descriptions}

Conversation history:
---
{formatted_history}
---

Respond with ONLY ONE key: {participant_options}
"""
        try:
            resp = self.moderator_llm.invoke(prompt)
            key = resp.content.strip().lower()
            if key in participants:
                print(f"Moderator selected: {key}")
                state["current_speaker"] = key
            else:
                print(f"Invalid selection '{key}', choosing randomly.")
                state["current_speaker"] = random.choice(list(participants.keys()))
        except Exception as e:
            print(f"Error selecting speaker: {e}. Choosing randomly.")
            state["current_speaker"] = random.choice(list(participants.keys()))
        return state
    def add_participant(self, key: str):
        """Register a participant key for interactive setup."""
        self.initial_participants.append(key)

    def start_conversation(self,
                           topic: str,
                           initial_participants: List[str] = None,
                           total_turns: int = None) -> List[AIMessage | HumanMessage]:
        """Initializes and runs the conversation graph."""
        # Handle interactive vs. demo calls
        # If only an int is passed as initial_participants, treat as total_turns
        if isinstance(initial_participants, int):
            total_turns = initial_participants
            participants_keys = self.initial_participants
        else:
            participants_keys = (initial_participants or self.initial_participants)
        if total_turns is not None:
            self.total_turns = total_turns
        # Build agents dict
        participants = {key: create_agent(key) for key in participants_keys}
        
        initial_state = ConversationState(
            participants=participants,
            chat_history=[HumanMessage(content=topic, name="Human")],
            current_speaker=random.choice(list(participants.keys())),  # Start with a random speaker
            turn_count=0,
            ontology_insights=[]  # Start with no insights
        )
        
        print(f"\nStarting ethical debate on: '{topic}'")
        print("=" * 60)
        
        # LangGraph's stream method executes the graph
        final_state = self.graph.invoke(initial_state)
        
        print("\n" + "=" * 60)
        print("Conversation complete!")
        return final_state['chat_history']