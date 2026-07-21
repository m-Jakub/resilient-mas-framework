"""
Unified per-turn execution pipeline for AEGIS.

Migration note:
- AegisOrchestrator delegates per-turn work to TurnController.
- Logging preserves legacy metadata keys while adding optional extensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import json
import string
import time

from langchain.schema import AIMessage

from src.kg_cfr.aegis_drau_context import AegisDrauContextBuilder
from src.kg_cfr.aegis_logger import AegisLogger
from src.kg_cfr.counterfactual_reflection import CounterfactualReflectionEngine
from src.common.ontology import ETHICAL_CONCEPTS


AEGIS_KEYWORDS = [
    "power", "grid", "reserve", "load", "blackout", "transformer", "telemetry",
    "hospital", "military", "periphery", "triage", "allocation", "substation",
]


@dataclass
class DebateState:
    crisis_context: str
    chat_history: List[Any]
    modules: Dict[str, Any]
    module_labels: Dict[str, str]
    turn_number: int
    debate_turns: int
    shock_event: Optional[Dict[str, Any]]
    cfr_mode: str
    use_id_rag: bool
    use_adaptive_idrag: bool
    cf_flags: Dict[str, Any] = field(default_factory=dict)
    private_thought_buffer: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_aegis_state(cls, state: Dict[str, Any], debate_turns: int) -> "DebateState":
        return cls(
            crisis_context=state.get("crisis_context", ""),
            chat_history=state.get("chat_history", []),
            modules=state.get("modules", {}),
            module_labels=state.get("module_labels", {}),
            turn_number=state.get("turn_count", 0) + 1,
            debate_turns=debate_turns,
            shock_event=state.get("shock_event"),
            cfr_mode=state.get("cfr_mode", "no_cfr_baseline"),
            use_id_rag=bool(state.get("use_id_rag")),
            use_adaptive_idrag=bool(state.get("use_adaptive_idrag")),
            cf_flags=state.get("cf_flags", {}),
            private_thought_buffer=state.get("private_thought_buffer", {}),
        )

    def apply_to_aegis_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        state["cf_flags"] = self.cf_flags
        state["private_thought_buffer"] = self.private_thought_buffer
        state["chat_history"] = self.chat_history
        state["modules"] = self.modules
        state["module_labels"] = self.module_labels
        return state


@dataclass
class PlannerOutput:
    private_strategy_json: str = ""
    private_strategy_nl: str = ""
    cfr_generated: bool = False
    cfr_injected: bool = False
    rag_used: bool = False
    rag_docs_count: int = 0
    cf_flags: Dict[str, Any] = field(default_factory=dict)
    buffer_payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KGResult:
    query: str
    retrieved_axioms: List[str]
    rag_docs_count: int
    status: str
    latency_ms: Optional[int] = None
    fallback_reason: str = ""


@dataclass
class TurnResult:
    response: str
    planner_output: PlannerOutput
    kg_result: Optional[KGResult]
    metadata: Dict[str, Any]


class StrategyPlanner:
    def __init__(
        self,
        cr_engine: Optional[CounterfactualReflectionEngine],
        debate_turns: int,
        cfr_mode: str,
    ) -> None:
        self.cr_engine = cr_engine
        self.debate_turns = debate_turns
        self.cfr_mode = cfr_mode

    def plan(
        self,
        debate_state: DebateState,
        module_id: str,
        last_message: str,
        attack_vector: str,
        shock_context: str,
        rag_gateway: Optional["KGGateway"],
    ) -> Tuple[PlannerOutput, Optional[KGResult]]:
        # 1. Base case: Jeśli tryb to baseline, zwracamy puste flagi
        if self.cfr_mode == "no_cfr_baseline" or not self.cr_engine:
            cf_flags = {"activated": False, "turn_idx": debate_state.turn_number - 1}
            return PlannerOutput(cf_flags=cf_flags), None

        # 2. Pobieramy gotowy wynik CFR wprost z Orchestratora
        cfr_result = debate_state.private_thought_buffer.get(module_id)
        kg_result = None

        if cfr_result:
            # Wydobycie metadanych z obiektu CounterfactualReflectionOutput
            rag_docs_count = getattr(cfr_result, "audit", {}).get("rag_docs_count", 0)
            guardrail_flags = getattr(cfr_result, "guardrail_flags", {})
            
            buffer_payload = {
                "scenario": getattr(cfr_result, "scenario", ""),
                "private_strategy_json": getattr(cfr_result, "strategic_insight", "{}"),
                "private_strategy": getattr(cfr_result, "nl_instruction", ""),
                "guardrail_flags": guardrail_flags,
                "rag_used": rag_docs_count > 0,
                "rag_docs_count": rag_docs_count,
            }
            
            cf_flags = {
                "activated": True,
                "turn_idx": debate_state.turn_number - 1,
                "cfr_mode": self.cfr_mode,
                **guardrail_flags,
            }
            
            planner = PlannerOutput(
                private_strategy_json=buffer_payload["private_strategy_json"],
                private_strategy_nl=buffer_payload["private_strategy"],
                cfr_generated=True,
                cfr_injected=bool(buffer_payload["private_strategy"]),
                rag_used=rag_docs_count > 0,
                rag_docs_count=rag_docs_count,
                cf_flags=cf_flags,
                buffer_payload=buffer_payload,
            )

            # Rekonstrukcja metryk RAG dla logów kompatybilnych z artykułem
            if self.cfr_mode == "kg_cfr_full":
                if rag_docs_count > 0:
                    kg_result = KGResult(query="orchestrator", retrieved_axioms=[], rag_docs_count=rag_docs_count, status="ok")
                else:
                    kg_result = KGResult(query="orchestrator", retrieved_axioms=[], rag_docs_count=0, status="empty", fallback_reason="no_evidence")

            return planner, kg_result

        # 3. Fallback: Jeśli CFR z jakiegoś powodu nie zwróciło wyniku w Orchestratorze
        cf_flags = {"activated": False, "turn_idx": debate_state.turn_number - 1}
        return PlannerOutput(cf_flags=cf_flags), None


class KGGateway:
    def __init__(self, module: Any) -> None:
        self.module = module

    def retrieve(self, query: str) -> KGResult:
        start = time.time()
        agent = getattr(self.module, "agent", None)
        identity_graph = getattr(agent, "identity_graph", None) if agent else None
        if not identity_graph:
            return KGResult(
                query=query,
                retrieved_axioms=[],
                rag_docs_count=0,
                status="disabled",
                latency_ms=int((time.time() - start) * 1000),
                fallback_reason="missing_identity_graph",
            )

        query_lc = (query or "").lower()
        matched = [k for k in list(ETHICAL_CONCEPTS.keys()) + AEGIS_KEYWORDS if k.lower() in query_lc]
        if matched:
            search_terms = matched
        else:
            stop_words = {
                "the", "a", "an", "is", "are", "to", "of", "in", "that", "for", "with",
                "on", "what", "how", "this", "it",
            }
            tokens = [t.strip(string.punctuation) for t in query_lc.split()]
            candidates = [(idx, t) for idx, t in enumerate(tokens) if t and t not in stop_words]
            candidates.sort(key=lambda item: (-len(item[1]), item[0]))
            search_terms = [term for _, term in candidates[:5]]
        evidence_nodes: List[Dict[str, Any]] = []
        for term in search_terms:
            evidence_nodes.extend(identity_graph.get_beliefs_about_concept(term))
        evidence_texts = [n.get("content", "") for n in evidence_nodes if n.get("content")]
        if not evidence_texts:
            core_beliefs = identity_graph.get_core_beliefs() or []
            evidence_texts = [
                b.get("content", "")
                for b in core_beliefs
                if b.get("content")
            ]

        latency_ms = int((time.time() - start) * 1000)
        status = "ok" if evidence_texts else "empty"
        fallback_reason = "no_evidence" if not evidence_texts else ""

        return KGResult(
            query=query,
            retrieved_axioms=evidence_texts[:8],
            rag_docs_count=len(evidence_texts[:8]),
            status=status,
            latency_ms=latency_ms,
            fallback_reason=fallback_reason,
        )


class ResponseComposer:
    def compose(
        self,
        module: Any,
        crisis_context: str,
        attack_vector: str,
        shock_context: str,
        chat_history: List[Any],
        private_strategy_nl: str,
        context: str,
        use_id_rag: bool,
        use_adaptive_idrag: bool,
    ) -> str:
        return module.respond(
            crisis_context=crisis_context,
            attack_vector=attack_vector,
            shock_context=shock_context,
            chat_history=chat_history,
            context=context,
            private_strategy=private_strategy_nl,
            use_id_rag=use_id_rag,
            use_adaptive_idrag=use_adaptive_idrag,
        )


class TurnLogger:
    def __init__(self, logger: Optional[AegisLogger]) -> None:
        self.logger = logger

    def build_metadata(
        self,
        planner_output: PlannerOutput,
        kg_result: Optional[KGResult],
        opponent_claims: List[str],
        profile_name: str,
        planner_version: str,
    ) -> Dict[str, Any]:
        cf_flags = dict(planner_output.cf_flags or {})
        if "activated" not in cf_flags:
            cf_flags["activated"] = False
        kg_status = kg_result.status if kg_result else "disabled"
        kg_latency_ms = kg_result.latency_ms if kg_result else None
        kg_fallback_reason = kg_result.fallback_reason if kg_result else ""

        metadata = {
            "cf_flags": cf_flags,
            "private_strategy_json": planner_output.private_strategy_json,
            "private_strategy_nl": planner_output.private_strategy_nl,
            "cfr_generated": planner_output.cfr_generated,
            "cfr_injected": planner_output.cfr_injected,
            "rag_used": planner_output.rag_used,
            "rag_docs_count": planner_output.rag_docs_count,
            "opponent_claims": opponent_claims,
            # Compatibility keys expected by downstream analysis
            "privatestrategyjson": planner_output.private_strategy_json,
            "cfrgenerated": planner_output.cfr_generated,
            "ragdocscount": planner_output.rag_docs_count,
            "cfflags": cf_flags,
            # Optional extensions
            "kg_status": kg_status,
            "kg_latency_ms": kg_latency_ms,
            "kg_fallback_reason": kg_fallback_reason,
            "planner_version": planner_version,
            "profile_name": profile_name,
        }
        return metadata

    def log_turn(
        self,
        turn_number: int,
        module_id: str,
        module_label: str,
        response: str,
        concepts: List[str],
        metadata: Dict[str, Any],
    ) -> None:
        if not self.logger:
            return
        self.logger.log_turn(
            turn_number=turn_number,
            phase="standoff",
            module_id=module_id,
            module_label=module_label,
            message_content=response,
            concepts_mentioned=concepts,
            metadata=metadata,
        )


class TurnController:
    def __init__(
        self,
        context_builder: AegisDrauContextBuilder,
        cr_engine: Optional[CounterfactualReflectionEngine],
        debate_turns: int,
        cfr_mode: str,
        logger: Optional[AegisLogger] = None,
        planner_version: str = "v1",
    ) -> None:
        self.context_builder = context_builder
        self.cr_engine = cr_engine
        self.debate_turns = debate_turns
        self.cfr_mode = cfr_mode
        self.planner_version = planner_version
        self.strategy_planner = StrategyPlanner(cr_engine, debate_turns, cfr_mode)
        self.response_composer = ResponseComposer()
        self.turn_logger = TurnLogger(logger)

    def _profile_name(self) -> str:
        if self.cfr_mode == "no_cfr_baseline":
            return "baseline"
        if self.cfr_mode == "cfr_no_kg":
            return "reasoning_only"
        if self.cfr_mode == "kg_cfr_full":
            return "reasoning_plus_kg"
        return "unknown"

    def run_turn(
        self,
        debate_state: DebateState,
        module_id: str,
        module: Any,
        module_label: str,
    ) -> TurnResult:
        attack_vector, opponent_claims = self.context_builder.build_attack_vector_with_claims(
            {
                "chat_history": debate_state.chat_history,
                "crisis_context": debate_state.crisis_context,
                "modules": debate_state.modules,
                "module_labels": debate_state.module_labels,
                "current_module_id": module_id,
                "shock_event": debate_state.shock_event,
            },
            module_id,
        )
        shock_context = ""
        if debate_state.shock_event:
            shock_context = debate_state.shock_event.get("text", "")

        last_message = attack_vector or (debate_state.chat_history[-1].content if debate_state.chat_history else "")
        kg_gateway = KGGateway(module) if self.cfr_mode == "kg_cfr_full" else None

        planner_output, kg_result = self.strategy_planner.plan(
            debate_state=debate_state,
            module_id=module_id,
            last_message=last_message,
            attack_vector=attack_vector,
            shock_context=shock_context,
            rag_gateway=kg_gateway,
        )

        override_id_rag = False if self.cfr_mode == "kg_cfr_full" else debate_state.use_id_rag
        override_adaptive = False if self.cfr_mode == "kg_cfr_full" else debate_state.use_adaptive_idrag

        response = self.response_composer.compose(
            module=module,
            crisis_context=debate_state.crisis_context,
            attack_vector=attack_vector,
            shock_context=shock_context,
            chat_history=debate_state.chat_history,
            private_strategy_nl=planner_output.private_strategy_nl,
            context="",
            use_id_rag=override_id_rag,
            use_adaptive_idrag=override_adaptive,
        )

        debate_state.chat_history.append(AIMessage(content=response, name=f"{module_id}:{module_label}"))
        debate_state.cf_flags = planner_output.cf_flags
        if planner_output.buffer_payload:
            debate_state.private_thought_buffer[module_id] = planner_output.buffer_payload
        else:
            debate_state.private_thought_buffer.pop(module_id, None)

        metadata = self.turn_logger.build_metadata(
            planner_output=planner_output,
            kg_result=kg_result,
            opponent_claims=opponent_claims,
            profile_name=self._profile_name(),
            planner_version=self.planner_version,
        )

        concepts = [
            k for k in list(ETHICAL_CONCEPTS.keys()) + AEGIS_KEYWORDS
            if k.lower() in (response or "").lower()
        ]
        self.turn_logger.log_turn(
            turn_number=debate_state.turn_number,
            module_id=module_id,
            module_label=module_label,
            response=response,
            concepts=sorted(set(concepts)),
            metadata=metadata,
        )

        return TurnResult(response=response, planner_output=planner_output, kg_result=kg_result, metadata=metadata)
