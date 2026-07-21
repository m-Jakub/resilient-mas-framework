"""
Baseline Tutor Agent for A/B Testing (Control Condition - Mode A)

This module implements a single objective expert-tutor LLM agent that:
- Has unified RAG access to ALL philosophical corpora (Socrates, Mill, Kant, Augustine, Aquinas)
- Presents multiple ethical perspectives objectively without debate
- Serves as the control condition for testing KQ1: Does debate improve learning?

Architecture:
    Single LLM Agent → Unified RAG (All Stores) → Objective Multi-Perspective Response
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config import LLM_MODEL, TEMPERATURE, SIMILARITY_SEARCH_K
import time


class BaselineTutorAgent:
    """
    Single objective expert-tutor with unified RAG across all philosophical traditions.
    
    This agent serves as the control condition (Mode A) in A/B testing, providing
    students with comprehensive multi-perspective analysis without debate structure.
    
    Key Features:
    - Unified retrieval from all philosophical stores (Socrates, Kant, Mill, Augustine, Aquinas)
    - Objective presentation of Deontology, Utilitarianism, Virtue Ethics, Christian Ethics
    - Conversational memory for follow-up questions
    - RAG-grounded responses with source citations
    """
    
    def __init__(self, 
                 chroma_dir: str = "data/chroma",
                 k: int = 6,
                 temperature: float = TEMPERATURE,
                 max_turns: int = 10):
        """
        Initialize baseline tutor with unified RAG access.
        
        Args:
            chroma_dir: Path to directory containing all Chroma vector stores
            k: Number of documents to retrieve per store
            temperature: LLM temperature (0.0-1.0)
            max_turns: Maximum conversation turns
        """
        self.chroma_dir = Path(chroma_dir)
        self.k = k
        self.temperature = temperature
        self.max_turns = max_turns
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=temperature,
            max_output_tokens=2048
        )
        
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize embeddings (same as used in debate mode for fair comparison)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",  # Must match vector store embeddings (1024 dim)
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity search
        )
        
        # Initialize unified retrievers
        self.retrievers = self._initialize_unified_retrievers()
        
        # Conversation memory
        self.chat_history: List[HumanMessage | AIMessage] = []
        
        print(f"[Baseline Tutor] Initialized with unified RAG access to {len(self.retrievers)} philosophical stores")
    
    def _initialize_unified_retrievers(self) -> Dict[str, object]:
        """
        Initialize retrievers for ALL available philosophical stores.
        
        Returns:
            Dictionary mapping store names to their retrievers
        """
        stores = {
            "socrates": "socrates_store",
            "mill": "mill_store", 
            "kant": "kant_store",
            "aquinas": "aquinas_store",
            "plato": "plato_store",
            "aristotle": "aristotle_store",
            "bentham": "bentham_store",
            "nietzsche": "nietzsche_store",
            "st_augustine": "st_augustine_store"
        }
        
        retrievers = {}
        
        for key, collection_name in stores.items():
            store_path = self.chroma_dir / collection_name
            
            if not store_path.exists():
                print(f"[Warning] Store not found: {store_path}")
                continue
            
            try:
                vectorstore = Chroma(
                    persist_directory=str(store_path),
                    embedding_function=self.embeddings,
                    collection_name=collection_name
                )
                
                retrievers[key] = vectorstore.as_retriever(
                    search_kwargs={"k": self.k}
                )
                
                print(f"[Baseline Tutor] Loaded RAG store: {key}")
                
            except Exception as e:
                print(f"[Error] Failed to load {key}: {e}")
        
        return retrievers
    
    def _retrieve_unified_context(self, query: str) -> Tuple[str, Dict[str, int]]:
        """
        Retrieve relevant passages from ALL philosophical stores.
        
        This ensures the baseline tutor has access to the same knowledge base
        as the collective debate agents, maintaining experimental fairness.
        
        Args:
            query: Search query (typically the ethical dilemma)
        
        Returns:
            Tuple of (formatted context string, retrieval metadata)
        """
        all_passages = []
        metadata = {}
        
        for store_name, retriever in self.retrievers.items():
            try:
                docs = retriever.get_relevant_documents(query)
                
                if docs:
                    # Format passages with source attribution
                    for doc in docs:
                        passage = doc.page_content.strip()
                        source = store_name.replace("_", " ").title()
                        all_passages.append(f"[{source}] {passage}")
                    
                    metadata[store_name] = len(docs)
                    
            except Exception as e:
                print(f"[RAG Error] Failed to retrieve from {store_name}: {e}")
                metadata[store_name] = 0
        
        # Combine all passages
        context = "\n\n".join(all_passages)
        
        total_docs = sum(metadata.values())
        print(f"[RAG] Retrieved {total_docs} documents across {len(metadata)} stores")
        
        return context, metadata
    
    def _build_system_prompt(self) -> str:
        """
        Construct SIMPLE system prompt simulating "student pastes dilemma into ChatGPT."
        
        This is the baseline: what would a student get from vanilla LLM interaction
        without specialized debate architecture, ToM, or ID-RAG?
        
        Keeps RAG for fair comparison (LLM has access to same knowledge base),
        but removes complex pedagogical structure.
        """
        return """"""
    
    def generate_response(self, user_message: str) -> Dict[str, any]:
        """
        Generate tutor response with unified RAG context.
        
        Args:
            user_message: Student's question or dilemma
        
        Returns:
            Dictionary containing response, RAG metadata, and conversation state
        """
        start_time = time.time()
        
        # Retrieve context from all stores
        context, retrieval_metadata = self._retrieve_unified_context(user_message)
        
        # Build prompt
        system_prompt = self._build_system_prompt()
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("system", "**Context from Philosophical Texts:**\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        # Create chain
        chain = (
            RunnablePassthrough.assign(
                context=lambda x: context,
                chat_history=lambda x: self.chat_history
            )
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        # Generate response
        try:
            response = chain.invoke({"question": user_message})
            
            # Update conversation memory
            self.chat_history.append(HumanMessage(content=user_message))
            self.chat_history.append(AIMessage(content=response))
            
            elapsed = time.time() - start_time
            
            return {
                "response": response,
                "rag_metadata": {
                    "retrieved_docs": sum(retrieval_metadata.values()),
                    "stores_used": retrieval_metadata,
                    "context_length": len(context)
                },
                "turn_count": len(self.chat_history) // 2,
                "elapsed_seconds": round(elapsed, 2)
            }
            
        except Exception as e:
            print(f"[Error] Response generation failed: {e}")
            return {
                "response": "I apologize, but I encountered an error processing your question. Please try rephrasing.",
                "rag_metadata": {"error": str(e)},
                "turn_count": len(self.chat_history) // 2,
                "elapsed_seconds": 0
            }
    
    def get_conversation_log(self) -> List[Dict[str, str]]:
        """
        Export conversation history in structured format for analysis.
        
        Returns:
            List of turn dictionaries with speaker and message
        """
        log = []
        
        for i, message in enumerate(self.chat_history):
            log.append({
                "turn": (i // 2) + 1,
                "speaker": "Student" if isinstance(message, HumanMessage) else "Tutor",
                "message": message.content,
                "timestamp": time.time()  # Simplified; could track actual timestamps
            })
        
        return log
    
    def reset_conversation(self):
        """Clear conversation history for new session."""
        self.chat_history = []
        print("[Baseline Tutor] Conversation reset")


def run_baseline_tutor(dilemma: str, max_turns: int = 10) -> Dict[str, any]:
    """
    Main entry point for running baseline tutor mode (Mode A).
    
    This function orchestrates the baseline tutor interaction for A/B testing.
    
    Args:
        dilemma: Ethical dilemma to discuss
        max_turns: Maximum number of conversational turns
    
    Returns:
        Dictionary containing full interaction log and metadata
    """
    print("=" * 80)
    print("BASELINE TUTOR MODE (Control Condition - Mode A)")
    print("=" * 80)
    print(f"Dilemma: {dilemma}")
    print(f"Max turns: {max_turns}")
    print("=" * 80)
    
    # Initialize tutor
    tutor = BaselineTutorAgent(max_turns=max_turns)
    
    # Start with dilemma presentation
    print("\n[Student] ", dilemma)
    
    result = tutor.generate_response(dilemma)
    
    print(f"\n[Tutor] {result['response']}")
    if "error" in result['rag_metadata']:
        print(f"\n[API Error] Could not retrieve RAG stats due to LLM failure: {result['rag_metadata']['error']}")
    else:
        print(f"\n[RAG] Retrieved {result['rag_metadata']['retrieved_docs']} documents")
        print(f"[RAG] Stores used: {result['rag_metadata']['stores_used']}")
    
    # Interactive follow-up loop (for testing; in experiment, pre-scripted or student-driven)
    turn_count = 1
    
    while turn_count < max_turns:
        print(f"\n--- Turn {turn_count + 1}/{max_turns} ---")
        
        # In actual experiment, this would come from student input or pre-test protocol
        follow_up = input("\n[Student] Your question (or 'quit' to end): ").strip()
        
        if follow_up.lower() in ['quit', 'exit', 'q', '']:
            print("\n[System] Ending baseline tutor session.")
            break
        
        result = tutor.generate_response(follow_up)
        
        print(f"\n[Tutor] {result['response']}")
        if "error" in result['rag_metadata']:
            print(f"[API Error] Details: {result['rag_metadata']['error']}")
        else:
            print(f"[RAG] Retrieved {result['rag_metadata']['retrieved_docs']} documents")
        
        turn_count += 1
    
    # Export interaction log
    conversation_log = tutor.get_conversation_log()
    
    final_metadata = {
        "mode": "baseline",
        "dilemma": dilemma,
        "total_turns": len(conversation_log) // 2,
        "max_turns": max_turns,
        "conversation_log": conversation_log
    }
    
    print("\n" + "=" * 80)
    print(f"BASELINE SESSION COMPLETE: {final_metadata['total_turns']} turns")
    print("=" * 80)
    
    return final_metadata


if __name__ == "__main__":
    # Smoke test
    test_dilemma = (
        "The trolley problem: A runaway trolley is headed towards five people. "
        "You can pull a lever to divert it, killing one person instead. "
        "What is the ethical course of action?"
    )
    
    run_baseline_tutor(test_dilemma, max_turns=3)
