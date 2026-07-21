"""
Socratic Moderator Module for Multi-Agent Philosophical Debates

This module implements a RAG-enhanced Socratic-style moderator that intervenes 
during team deliberations and debates to promote deeper reflection, clarify concepts, 
and expose hidden assumptions using philosophically grounded questions.

Architecture:
    Team Deliberation → RAG Retrieval → Moderator Intervention (Socratic Questions) → 
    Refined Understanding → Debate

The moderator operates between deliberation and debate phases, analyzing team
discussions for opportunities to deepen philosophical inquiry. It uses Retrieval-
Augmented Generation (RAG) to ground questions in actual philosophical texts.

RAG Integration:
    - Retrieves relevant philosophical passages from knowledge base
    - Uses retrieved context to generate informed Socratic questions
    - References classical arguments and concepts from philosophical tradition
"""

import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import yaml
from pathlib import Path
import random

from src.hde.team import Team
from src.hde.enhanced_ontology import EnhancedOntology, ENHANCED_ONTOLOGY
from src.common.philosopher_agents import _rate_limit_wait  # Import rate limiting helper
from src.common.api_abstraction import get_philosopher_api  # Use API abstraction
from src.common.llm_retry import invoke_with_retry
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from config.settings import LLM_MODEL


@dataclass
class ModerationRecord:
    """Record of a single moderation intervention with RAG context."""
    timestamp: str
    team_id: str
    question_type: str
    question: str
    context: str
    team_response: Optional[str] = None
    intervention_reason: str = ""
    retrieved_sources: List[str] = field(default_factory=list)  # RAG-retrieved passages
    rag_grounded: bool = False  # Whether question used RAG context


class SocraticModerator:
    """
    RAG-enhanced Socratic-style moderator that intervenes in philosophical debates.
    
    The moderator uses Socratic questioning techniques enhanced with RAG:
    - Clarify vague or ambiguous concepts
    - Expose hidden assumptions
    - Test consistency of arguments
    - Deepen philosophical inquiry with references to philosophical texts
    - Bridge conceptual gaps using classical arguments
    
    RAG Capabilities:
    - Retrieves relevant philosophical passages from vector store
    - Grounds questions in actual philosophical arguments
    - References Kantian, Utilitarian, Aristotelian, and Christian ethics sources
    
    Attributes:
        question_bank: Templates for Socratic questions by category
        tone_profile: Conversational tone (e.g., "reflective", "inquisitive")
        conversation_log: History of all moderation interventions
        ontology: Reference to ethical concepts ontology
        intervention_threshold: Minimum score to trigger intervention
        rag_client: Vector store retriever for philosophical texts (optional)
        enable_rag: Whether to use RAG for question generation
    """
    
    def __init__(
        self,
        prompts_path: Optional[Path] = None,
        tone_profile: str = "reflective",
        ontology: Optional[EnhancedOntology] = None,
        intervention_threshold: float = 0.6,
        rag_client: Optional[Any] = None,
        enable_rag: bool = True,
        llm: Optional[Any] = None
    ):
        """
        Initialize the RAG-enhanced Socratic moderator.
        
        Args:
            prompts_path: Path to YAML file with question templates
            tone_profile: Conversational tone ("reflective", "inquisitive", "challenging")
            ontology: Ethical concepts ontology for semantic analysis
            intervention_threshold: Score threshold (0-1) for triggering intervention
            rag_client: Optional RAG retriever (Chroma, FAISS, or custom)
            enable_rag: Whether to enable RAG-enhanced questioning
            llm: Optional LLM for authentic Socratic question generation
        """
        self.tone_profile = tone_profile
        self.conversation_log: List[ModerationRecord] = []
        self.ontology = ontology or ENHANCED_ONTOLOGY
        self.intervention_threshold = intervention_threshold
        self.enable_rag = enable_rag
        self.rag_client = rag_client
        
        # Initialize LLM for question generation using API abstraction
        if llm is None:
            try:
                api = get_philosopher_api()
                self.llm = api.get_llm()
            except Exception as e:
                print(f"[Warning] LLM initialization failed: {e}")
                self.llm = None
        else:
            self.llm = llm
        
        # Initialize RAG client if not provided but enabled
        if self.enable_rag and self.rag_client is None:
            self.rag_client = self._initialize_default_rag_client()
        
        # Load question bank from YAML
        if prompts_path is None:
            prompts_path = Path(__file__).parent.parent / "config" / "moderation_prompts.yaml"
        
        self.question_bank = self._load_question_bank(prompts_path)
        
        # Track which questions have been used (avoid repetition)
        self._used_questions: Dict[str, List[str]] = {
            category: [] for category in self.question_bank.keys()
        }
        
        # SysAR perturbations moved to SystemDispatcher (S1-S3)
    
    def _initialize_default_rag_client(self) -> Optional[Dict[str, Any]]:
        """
        Initialize default RAG clients using existing Chroma stores.
        
        Returns:
            Dictionary mapping philosophical schools to their retrievers
        """
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Load existing Chroma stores
            data_dir = Path(__file__).parent.parent / "data" / "chroma"
            
            rag_stores = {}
            philosophical_schools = [
                "socrates_store",          # Platonic dialogues (highest priority)
                "mill_store",
                "kant_store", 
                "aquinas_store",
                "plato_store",
                "aristotle_store",
                "bentham_store",
                "nietzsche_store",
                "st_augustine_store"
            ]
            
            for school in philosophical_schools:
                store_path = data_dir / school
                if store_path.exists():
                    try:
                        vectorstore = Chroma(
                            persist_directory=str(store_path),
                            embedding_function=embeddings
                        )
                        rag_stores[school] = vectorstore.as_retriever(
                            search_kwargs={"k": 6}  # Increased from 3 to 6
                        )
                    except Exception as e:
                        print(f"[Warning] Could not load {school}: {e}")
            
            return rag_stores if rag_stores else None
            
        except Exception as e:
            print(f"[Warning] RAG initialization failed: {e}")
            return None
    
    def _build_rag_query(
        self,
        topic: str,
        team_concepts: List[str],
        last_team_message: Optional[str] = None
    ) -> str:
        """
        Build a rich, contextual query for RAG retrieval.
        
        Strategy:
        1. Start with the discussion topic
        2. Add top-3 ethical concepts from team memory
        3. Optionally include snippet of last team message for context
        4. Normalize text (lowercase, remove special chars)
        5. Limit to 512 characters
        
        Args:
            topic: Current discussion topic/dilemma
            team_concepts: List of ethical concepts from team memory
            last_team_message: Optional last message from team for context
            
        Returns:
            Normalized query string (max 512 chars)
        """
        # Start with topic
        query_parts = [topic]
        
        # Add top-3 concepts if available
        if team_concepts:
            top_concepts = team_concepts[:3]
            concepts_str = ", ".join(top_concepts)
            query_parts.append(f"Key concepts: {concepts_str}")
        
        # Add recent context if available
        if last_team_message:
            # Take first 150 chars of last message
            recent_context = last_team_message[:150].strip()
            query_parts.append(f"Recent discussion: {recent_context}")
        
        # Join parts
        raw_query = ". ".join(query_parts)
        
        # Normalize: lowercase and remove excessive special chars
        normalized = raw_query.lower()
        # Keep only alphanumeric, spaces, and basic punctuation
        import re
        normalized = re.sub(r'[^\w\s\.\,\:\;\?\!-]', ' ', normalized)
        # Collapse multiple spaces
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Limit to 512 chars
        if len(normalized) > 512:
            normalized = normalized[:512].rsplit(' ', 1)[0]  # Cut at word boundary
        
        return normalized
    
    def _load_question_bank(self, path: Path) -> Dict[str, List[str]]:
        """
        Load Socratic question templates from YAML file.
        
        Args:
            path: Path to YAML file with question templates
            
        Returns:
            Dictionary mapping question categories to template lists
        """
        if not path.exists():
            # Return default question bank if file doesn't exist
            return self._get_default_question_bank()
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data.get('questions', self._get_default_question_bank())
        except Exception as e:
            print(f"[Warning] Failed to load prompts from {path}: {e}")
            return self._get_default_question_bank()
    
    def _get_default_question_bank(self) -> Dict[str, List[str]]:
        """Return default Socratic question templates."""
        return {
            "clarification": [
                "What do you mean by '{concept}'?",
                "Could you clarify what you understand by '{concept}'?",
                "When you say '{concept}', what exactly do you have in mind?",
                "How are you defining '{concept}' in this context?",
            ],
            "counterfactual": [
                "What if the principle of '{concept}' were reversed?",
                "How would your argument change if '{concept}' were not the case?",
                "What would happen if we applied '{concept}' universally?",
                "Consider the opposite: what if '{concept}' led to harm?",
            ],
            "definitional": [
                "How do we define '{concept}' in this context?",
                "What are the essential characteristics of '{concept}'?",
                "Is '{concept}' the same as '{related_concept}', or are they distinct?",
                "What distinguishes '{concept}' from '{opposed_concept}'?",
            ],
            "epistemic": [
                "How do we know that '{concept}' is true or valid?",
                "What evidence supports the claim about '{concept}'?",
                "On what grounds can we assert that '{concept}' is overriding?",
                "What justifies prioritizing '{concept}' over other values?",
            ],
            "comparative": [
                "How does '{concept}' compare to '{opposed_concept}'?",
                "What would a proponent of '{opposed_concept}' say about this?",
                "Can '{concept}' and '{related_concept}' be reconciled?",
                "Which framework better accounts for '{concept}'—yours or theirs?",
            ],
            "consistency": [
                "Is your use of '{concept}' consistent with your earlier claims?",
                "How does this view of '{concept}' align with your team's position?",
                "Does applying '{concept}' here contradict your previous argument?",
                "Can you maintain both '{concept}' and '{opposed_concept}' without contradiction?",
            ],
            "implication": [
                "What are the implications of accepting '{concept}'?",
                "If '{concept}' is true, what follows from that?",
                "Does commitment to '{concept}' require accepting certain consequences?",
                "What else must be true if '{concept}' holds?",
            ]
        }
    
    def fetch_contextual_passages(
        self,
        topic: str,
        concepts: List[str],
        last_team_message: Optional[str] = None,
        k: int = 6
    ) -> Tuple[List[str], str]:
        """
        Retrieve philosophically relevant passages from RAG knowledge base.
        
        This method performs semantic search over philosophical texts to find
        passages relevant to the current discussion topic and ethical concepts.
        
        Enhanced features:
        - Rich query building with topic + concepts + context
        - Text normalization for better matching
        - Increased k for more diverse results
        - Debug logging of query and retrieval stats
        
        Args:
            topic: Current discussion topic/dilemma
            concepts: List of ethical concepts being discussed
            last_team_message: Optional recent message for context
            k: Number of passages to retrieve (default: 6, increased from 3)
            
        Returns:
            Tuple of (retrieved_passages, query_used)
        """
        if not self.enable_rag or not self.rag_client:
            return ([], "")
        
        # Build rich query using new method
        search_query = self._build_rag_query(
            topic=topic,
            team_concepts=concepts,
            last_team_message=last_team_message
        )
        
        print(f"\n[RAG] Query: {search_query[:200]}...")  # Debug log
        
        retrieved_passages = []
        
        try:
            # If rag_client is a dict of retrievers (multiple stores)
            if isinstance(self.rag_client, dict):
                # Strategy: Try to match concepts to appropriate philosophical stores
                matched_stores = []
                
                # Map concepts to store names
                concept_store_mapping = {
                    # Kant - Deontology
                    "duty": "kant_store",
                    "categorical imperative": "kant_store",
                    "autonomy": "kant_store",
                    "moral law": "kant_store",
                    # Mill - Utilitarianism
                    "utility": "mill_store",
                    "happiness": "mill_store",
                    "consequence": "mill_store",
                    "pleasure": "mill_store",
                    "harm principle": "mill_store",
                    # Bentham - Classical Utilitarianism
                    "hedonic calculus": "bentham_store",
                    "pain": "bentham_store",
                    "greatest happiness": "bentham_store",
                    # Aquinas - Christian Ethics
                    "virtue": "aquinas_store",
                    "natural law": "aquinas_store",
                    "beatitude": "aquinas_store",
                    "grace": "aquinas_store",
                    # Aristotle - Virtue Ethics
                    "eudaimonia": "aristotle_store",
                    "phronesis": "aristotle_store",
                    "mean": "aristotle_store",
                    "flourishing": "aristotle_store",
                    # Plato - Forms
                    "form": "plato_store",
                    "the good": "plato_store",
                    "soul": "plato_store",
                    "justice": "plato_store",
                    # Nietzsche - Genealogy
                    "master morality": "nietzsche_store",
                    "slave morality": "nietzsche_store",
                    "will to power": "nietzsche_store",
                    "übermensch": "nietzsche_store",
                    # Augustine - Christian Patristic
                    "city of god": "st_augustine_store",
                    "original sin": "st_augustine_store",
                    "divine grace": "st_augustine_store",
                    # Socrates - Dialectic (fallback for general ethics)
                    "knowledge": "socrates_store",
                    "examined life": "socrates_store"
                }
                
                # Find relevant stores based on concepts
                for concept in concepts:
                    concept_lower = concept.lower()
                    if concept_lower in concept_store_mapping:
                        store_name = concept_store_mapping[concept_lower]
                        if store_name in self.rag_client and store_name not in matched_stores:
                            matched_stores.append(store_name)
                
                # If no matches, use all available stores (diverse retrieval)
                if not matched_stores:
                    matched_stores = list(self.rag_client.keys())
                
                print(f"[RAG] Searching stores: {matched_stores}")  # Debug log
                
                # Retrieve from matched stores
                for store_name in matched_stores[:2]:  # Limit to top 2 stores
                    try:
                        retriever = self.rag_client[store_name]
                        docs = retriever.get_relevant_documents(search_query)
                        
                        for doc in docs[:k//2]:  # Split k across stores
                            if hasattr(doc, 'page_content') and doc.page_content:
                                # Extract metadata if available
                                source = doc.metadata.get('source', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
                                passage = f"[{source}] {doc.page_content[:400]}"
                                retrieved_passages.append(passage)
                        
                        print(f"[RAG] Retrieved {len(docs)} docs from {store_name}")
                    except Exception as e:
                        print(f"[RAG Warning] Retrieval from {store_name} failed: {e}")
            
            # If rag_client is a single retriever
            elif hasattr(self.rag_client, 'get_relevant_documents'):
                docs = self.rag_client.get_relevant_documents(search_query)
                for doc in docs[:k]:
                    if hasattr(doc, 'page_content') and doc.page_content:
                        source = doc.metadata.get('source', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
                        passage = f"[{source}] {doc.page_content[:400]}"
                        retrieved_passages.append(passage)
                
                print(f"[RAG] Retrieved {len(docs)} total documents")
        
        except Exception as e:
            print(f"[RAG Warning] Retrieval failed: {e}")
            return ([], search_query)
        
        print(f"[RAG] Final passages count: {len(retrieved_passages)}")
        return (retrieved_passages[:k], search_query)
    
    def intervene_in_discussion(
        self,
        team: Team,
        topic: str,
        phase: str = "post-deliberation"
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze team discussion and decide whether to intervene with RAG-enhanced question.
        
        Flow:
        1. Assess if intervention is needed
        2. If yes, extract main ethical concepts
        3. Retrieve relevant philosophical passages via RAG (with rich query)
        4. Generate Socratic question grounded in retrieved context
        
        Args:
            team: Team object with memory and discussion history
            topic: Current discussion topic
            phase: Current phase ("post-deliberation", "mid-debate")
            
        Returns:
            Dictionary with intervention data (question_type, question, rag_metadata) 
            if intervention needed, else None
        """
        # Analyze team state to determine if intervention is needed
        intervention_score, reason = self._assess_intervention_need(team, topic)
        
        if intervention_score < self.intervention_threshold:
            return None
        
        # Extract main ethical concepts for RAG retrieval
        concepts = list(team.memory.ethical_concepts)
        
        # Get last message from team for context (if available)
        last_message = None
        if hasattr(team.memory, 'insights') and team.memory.insights:
            last_message = team.memory.insights[-1]
        
        # Fetch contextual passages from RAG if enabled (returns tuple now)
        # Always try RAG if enabled - even without concepts, topic alone is useful
        retrieved_passages = []
        rag_query = ""
        if self.enable_rag:
            retrieved_passages, rag_query = self.fetch_contextual_passages(
                topic=topic,
                concepts=concepts if concepts else [],
                last_team_message=last_message,
                k=6  # Increased from 3
            )
        
        # Generate appropriate Socratic question (with RAG context if available)
        question_type, question = self.generate_socratic_question(
            team, 
            topic, 
            reason,
            retrieved_context=retrieved_passages
        )
        
        # Prepare RAG metadata for logging
        rag_metadata = {
            "rag_query": rag_query,
            "rag_docs_count": len(retrieved_passages),
            "rag_sources": [p[:100] for p in retrieved_passages[:2]],  # Top-2 sources (truncated)
            "intervention_score": intervention_score,
            "intervention_reason": reason
        }
        
        # Log the intervention with RAG context (internal log)
        record = ModerationRecord(
            timestamp=datetime.now().isoformat(),
            team_id=team.team_id,
            question_type=question_type,
            question=question,
            context=f"Phase: {phase}, Topic: {topic[:100]}...",
            intervention_reason=reason,
            retrieved_sources=retrieved_passages,
            rag_grounded=bool(retrieved_passages)
        )
        self.conversation_log.append(record)
        
        # Return structured data for external logging
        return {
            "question_type": question_type,
            "question": question,
            "rag_metadata": rag_metadata
        }
    
    def _assess_intervention_need(
        self,
        team: Team,
        topic: str
    ) -> Tuple[float, str]:
        """
        Assess whether moderator intervention is needed.
        
        Intervention triggers:
        1. Team drifts off-topic (semantic distance too large)
        2. Internal contradictions detected (conflicting concepts)
        3. Epistemic gaps (unsupported claims)
        4. Insufficient conceptual depth
        
        Args:
            team: Team to assess
            topic: Discussion topic
            
        Returns:
            Tuple of (intervention_score, reason) where score is 0-1
        """
        scores = []
        reasons = []
        
        # Check 1: Conceptual conflicts (contradictions)
        ethical_concepts = list(team.memory.ethical_concepts)
        if len(ethical_concepts) >= 2:
            conflicts = self.ontology.analyze_concept_conflict(ethical_concepts)
            conflict_score = min(conflicts['conflict_count'] / 5.0, 1.0)  # Normalize
            if conflict_score > 0.4:
                scores.append(conflict_score)
                reasons.append(f"Conceptual conflict detected: {conflicts['conflict_count']} tensions")
        
        # Check 2: Insufficient concept usage (shallow discussion)
        if len(ethical_concepts) < 2:
            scores.append(0.8)
            reasons.append("Insufficient conceptual depth—only few concepts discussed")
        
        # Check 3: Repeated concepts without elaboration
        concept_usage = self.ontology.get_usage_summary()
        if concept_usage.get('most_used'):
            most_used_count = concept_usage['usage_by_concept'].get(
                concept_usage['most_used'], 0
            )
            if most_used_count > 5 and len(ethical_concepts) < 3:
                scores.append(0.7)
                reasons.append(f"Over-reliance on '{concept_usage['most_used']}' without exploring alternatives")
        
        # Check 4: Epistemic gaps (assertions without justification)
        # Check for claims in insights that lack supporting concepts
        insights_count = len(team.memory.insights)
        concepts_count = len(ethical_concepts)
        if insights_count > 3 and concepts_count < 2:
            scores.append(0.75)
            reasons.append("Many claims but insufficient conceptual grounding")
        
        # Calculate overall intervention score
        if not scores:
            return (0.0, "No intervention needed")
        
        avg_score = sum(scores) / len(scores)
        primary_reason = reasons[scores.index(max(scores))]
        
        return (avg_score, primary_reason)
    
    def generate_socratic_question(
        self,
        team: Team,
        topic: str,
        reason: str,
        retrieved_context: Optional[List[str]] = None
    ) -> Tuple[str, str]:
        """
        Generate a RAG-enhanced, context-appropriate Socratic question.
        
        If retrieved_context is provided, the question will reference or build
        upon philosophical ideas from the retrieved passages.
        
        Args:
            team: Team receiving the question
            topic: Discussion topic
            reason: Reason for intervention
            retrieved_context: Optional list of retrieved philosophical passages
            
        Returns:
            Tuple of (question_type, formatted_question)
        """
        # Select question category based on intervention reason
        category = self._select_question_category(reason)
        
        # PRIORITY 1: Try LLM-based Socratic generation if available
        if self.llm and retrieved_context:
            try:
                llm_question = self._generate_socratic_question_with_llm(
                    team, topic, category, retrieved_context, reason
                )
                if llm_question and len(llm_question) > 20:  # Sanity check
                    print(f"[Socrates LLM] Generated elenctic question")
                    return (category, llm_question)
            except Exception as e:
                print(f"[Warning] LLM question generation failed: {e}")
        
        # PRIORITY 2: Try RAG-grounded template-based question
        if retrieved_context and self.enable_rag:
            rag_question = self._generate_rag_grounded_question(
                team, topic, category, retrieved_context
            )
            if rag_question:
                return (category, rag_question)
        
        # PRIORITY 3: Fallback to template-based question
        # Get available questions in this category
        available_questions = [
            q for q in self.question_bank.get(category, [])
            if q not in self._used_questions[category]
        ]
        
        # If all questions used, reset
        if not available_questions:
            self._used_questions[category] = []
            available_questions = self.question_bank[category]
        
        # Select a question template
        template = random.choice(available_questions)
        self._used_questions[category].append(template)
        
        # Fill in the template with context
        question = self._contextualize_question(template, team, topic)
        
        return (category, question)
    
    def _generate_socratic_question_with_llm(
        self,
        team: Team,
        topic: str,
        category: str,
        retrieved_passages: List[str],
        reason: str
    ) -> Optional[str]:
        """
        Generate an authentic Socratic question using LLM with elenctic method.
        
        Uses retrieved philosophical passages as grounding for a dialectical question
        that challenges the team's reasoning through maieutics and aporia.
        
        Args:
            team: Team object with memory/summary
            topic: Current discussion topic
            category: Question category (assumption, reasoning, counterexample, etc.)
            retrieved_passages: List of relevant philosophical passages from RAG
            reason: Why intervention is needed (from ontology analysis)
            
        Returns:
            Generated Socratic question string, or None if generation fails
        """
        try:
            # Build team context from memory
            # TeamMemory has insights (List[str]), not dict with 'summary'
            team_summary = "ongoing discussion"
            if hasattr(team.memory, 'insights') and team.memory.insights:
                # Use last 2 insights as summary (each truncated to 150 chars)
                recent_insights = team.memory.insights[-2:]
                truncated = [insight[:150] for insight in recent_insights]
                team_summary = " ".join(truncated)
            elif hasattr(team.memory, 'key_arguments') and team.memory.key_arguments:
                # Fallback to key arguments
                team_summary = team.memory.key_arguments[-1][:150]
            
            # Format retrieved passages (limit to top 3, truncate each to 300 chars)
            passages_text = "\n".join([
                f"- {passage[:300]}..." if len(passage) > 300 else f"- {passage}"
                for passage in retrieved_passages[:3]
            ])
            
            # Build comprehensive Socratic prompt
            prompt = f"""You are Socrates, the ancient Greek philosopher known for the elenctic method (ἔλεγχος).

Your role is to moderate a philosophical discussion by asking questions that:
1. **Maieutics** (μαιευτική): Help participants "give birth" to their own insights by drawing out what they already know
2. **Aporia** (ἀπορία): Create productive puzzlement that reveals gaps in reasoning
3. **Dialectical inquiry**: Guide examination through careful questioning, not assertion

**Current Discussion Context:**
- Topic: {topic}
- Team's position: {team_summary[:200]}
- Identified issue: {reason}
- Question type needed: {category}

**Relevant Philosophical Passages Retrieved:**
{passages_text}

**Your Task:**
Generate ONE Socratic question (1-2 sentences max) that:
- References at least one of the passages above
- Challenges the team's reasoning in an open-ended way (no yes/no questions)
- Creates aporia by exposing an assumption or inconsistency
- Uses the elenctic method to guide them toward deeper understanding
- Remains philosophically grounded, not rhetorical

**Example of good Socratic questions:**
- "If we accept Kant's claim that persons cannot be mere means, how might this challenge the calculation you just proposed?"
- "You argue utility justifies this action—but does Mill's harm principle not suggest we examine who bears the cost?"

**Your question:**"""

            # Rate limiting: Wait if needed to avoid 429 errors
            _rate_limit_wait()

            # Generate with LLM
            response = invoke_with_retry(
                self.llm.invoke,
                prompt,
                operation="socrates_moderator.llm.invoke",
            )
            generated_question = response.content.strip()
            
            # Basic validation
            if len(generated_question) < 20:
                print(f"[Socrates LLM] Question too short ({len(generated_question)} chars), rejecting")
                return None
            
            if generated_question.startswith(("Your question:", "**Your question:**")):
                # Strip meta-text if LLM included it
                generated_question = generated_question.split(":", 1)[1].strip()
            
            return generated_question
            
        except Exception as e:
            print(f"[Socrates LLM] Error generating question: {e}")
            return None
    
    def _generate_rag_grounded_question(
        self,
        team: Team,
        topic: str,
        category: str,
        retrieved_passages: List[str]
    ) -> Optional[str]:
        """
        Generate a Socratic question grounded in retrieved philosophical texts.
        
        This method analyzes retrieved passages and crafts a question that
        references or builds upon the philosophical ideas found.
        
        Example:
            Retrieved: "Kant argues that moral worth comes from duty, not consequences."
            Question: "If moral worth stems from duty alone, can consequences ever 
                      override our obligations?"
        
        Args:
            team: Team context
            topic: Discussion topic
            category: Question category (epistemic, comparative, etc.)
            retrieved_passages: List of retrieved text excerpts
            
        Returns:
            RAG-grounded question, or None if generation fails
        """
        if not retrieved_passages:
            return None
        
        try:
            # Extract key philosophical claims from retrieved passages
            # Look for argumentative patterns
            concepts = list(team.memory.ethical_concepts)
            primary_concept = concepts[0] if concepts else "ethics"
            
            # Find passage most relevant to team's concepts
            best_passage = retrieved_passages[0]
            for passage in retrieved_passages:
                if any(c in passage.lower() for c in concepts):
                    best_passage = passage
                    break
            
            # Extract a key claim or principle (first 200 chars for brevity)
            philosophical_claim = best_passage[:200].strip()
            
            # Generate category-specific question referencing the claim
            if category == "epistemic":
                question = f"Drawing on classical arguments: if {philosophical_claim[:100]}..., how do we justify this claim in the current context?"
            elif category == "comparative":
                question = f"Classical texts suggest: {philosophical_claim[:100]}... How does this compare to your team's position?"
            elif category == "consistency":
                question = f"Consider this philosophical principle: {philosophical_claim[:100]}... Is your argument consistent with this view?"
            elif category == "definitional":
                question = f"Philosophical tradition holds: {philosophical_claim[:100]}... How do you define the key concept here?"
            elif category == "implication":
                question = f"If we accept: {philosophical_claim[:100]}..., what follows for your position?"
            else:  # clarification, counterfactual
                question = f"Reflecting on: {philosophical_claim[:100]}..., could you clarify your stance?"
            
            return question
            
        except Exception as e:
            print(f"[Debug] RAG question generation failed: {e}")
            return None
    
    def _select_question_category(self, reason: str) -> str:
        """
        Select appropriate question category based on intervention reason.
        
        Args:
            reason: Reason for intervention
            
        Returns:
            Question category name
        """
        reason_lower = reason.lower()
        
        if "conflict" in reason_lower or "contradiction" in reason_lower:
            return "consistency"
        elif "insufficient" in reason_lower or "shallow" in reason_lower:
            return "definitional"
        elif "over-reliance" in reason_lower:
            return "comparative"
        elif "grounding" in reason_lower or "justification" in reason_lower:
            return "epistemic"
        else:
            # Default to clarification
            return "clarification"
    
    def _contextualize_question(
        self,
        template: str,
        team: Team,
        topic: str
    ) -> str:
        """
        Fill question template with context-specific concepts.
        
        Args:
            template: Question template with placeholders
            team: Team context
            topic: Discussion topic
            
        Returns:
            Contextualized question
        """
        # Get concepts from team memory
        concepts = list(team.memory.ethical_concepts)
        
        # Get a primary concept to focus on
        concept = concepts[0] if concepts else "this principle"
        
        # Get related and opposed concepts from ontology
        related_concept = "related principle"
        opposed_concept = "opposing principle"
        
        if concept in self.ontology.concept_relationships:
            relations = self.ontology.concept_relationships[concept]
            if relations['related']:
                related_concept = relations['related'][0]
            if relations['opposed']:
                opposed_concept = relations['opposed'][0]
        
        # Replace placeholders
        question = template.replace('{concept}', concept)
        question = question.replace('{related_concept}', related_concept)
        question = question.replace('{opposed_concept}', opposed_concept)
        
        return question
    
    def record_team_response(
        self,
        team_id: str,
        response: str
    ) -> None:
        """
        Record a team's response to a Socratic question.
        
        Args:
            team_id: ID of responding team
            response: Team's response text
        """
        # Find the most recent intervention for this team
        for record in reversed(self.conversation_log):
            if record.team_id == team_id and record.team_response is None:
                record.team_response = response[:200]  # Store first 200 chars
                break
    
    def summarize_moderation_round(self) -> str:
        """
        Summarize all moderation interventions in current session, including RAG usage.
        
        Returns:
            Formatted summary of moderation history with RAG statistics
        """
        if not self.conversation_log:
            return "No moderation interventions yet."
        
        # Count RAG-grounded questions
        rag_grounded_count = sum(1 for r in self.conversation_log if r.rag_grounded)
        
        summary_parts = [
            "=== Moderation Summary ===",
            f"Total interventions: {len(self.conversation_log)}",
            f"RAG-grounded questions: {rag_grounded_count}/{len(self.conversation_log)}",
            ""
        ]
        
        # Group by question type
        by_type: Dict[str, int] = {}
        for record in self.conversation_log:
            by_type[record.question_type] = by_type.get(record.question_type, 0) + 1
        
        summary_parts.append("Questions by type:")
        for qtype, count in sorted(by_type.items()):
            summary_parts.append(f"  • {qtype}: {count}")
        
        summary_parts.append("")
        summary_parts.append("Recent interventions:")
        
        # Show last 3 interventions
        for record in self.conversation_log[-3:]:
            rag_indicator = " [RAG]" if record.rag_grounded else ""
            summary_parts.append(f"  [{record.timestamp}] {record.team_id}{rag_indicator}")
            summary_parts.append(f"    Type: {record.question_type}")
            summary_parts.append(f"    Q: {record.question}")
            if record.rag_grounded and record.retrieved_sources:
                summary_parts.append(f"    Sources: {len(record.retrieved_sources)} philosophical passages")
            if record.team_response:
                summary_parts.append(f"    Response: {record.team_response[:100]}...")
            summary_parts.append("")
        
        return "\n".join(summary_parts)
    
    def get_moderation_insights(self) -> Dict[str, any]:
        """
        Generate insights from moderation session, including RAG usage statistics.
        
        Returns:
            Dictionary with moderation statistics, RAG usage, and insights
        """
        if not self.conversation_log:
            return {
                "total_interventions": 0,
                "teams_moderated": [],
                "most_common_question_type": None,
                "intervention_reasons": [],
                "rag_grounded_count": 0,
                "rag_enabled": self.enable_rag
            }
        
        teams = set(r.team_id for r in self.conversation_log)
        rag_grounded_count = sum(1 for r in self.conversation_log if r.rag_grounded)
        question_types = [r.question_type for r in self.conversation_log]
        reasons = [r.intervention_reason for r in self.conversation_log]
        
        most_common = max(set(question_types), key=question_types.count)
        
        return {
            "total_interventions": len(self.conversation_log),
            "teams_moderated": list(teams),
            "most_common_question_type": most_common,
            "intervention_reasons": reasons,
            "rag_enabled": self.enable_rag,
            "rag_grounded_count": rag_grounded_count,
            "rag_percentage": (rag_grounded_count / len(self.conversation_log)) * 100 if self.conversation_log else 0,
            "questions_asked": [
                {
                    "team": r.team_id, 
                    "type": r.question_type, 
                    "question": r.question,
                    "rag_grounded": r.rag_grounded,
                    "sources_count": len(r.retrieved_sources) if r.retrieved_sources else 0
                }
                for r in self.conversation_log
            ]
        }


def integrate_moderator_with_team(
    team: Team,
    moderator: SocraticModerator,
    question: str
) -> None:
    """
    Integrate moderator question into team memory.
    
    This helper function adds the moderator's question to the team's
    strategic notes for reference during debate.
    
    Args:
        team: Team to receive the question
        moderator: Moderator instance
        question: Socratic question to add
    """
    note = f"[MODERATOR] {question}"
    # strategic_notes is a dict, not a list - use set_strategic_note() or add to list
    if not hasattr(team.memory, 'moderator_questions'):
        team.memory.moderator_questions = []
    team.memory.moderator_questions.append(note)


# Export public API
__all__ = [
    'SocraticModerator',
    'ModerationRecord',
    'integrate_moderator_with_team'
]
