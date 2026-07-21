"""
AEGIS/DRAU orchestrator for the Tripartite Standoff.

Replaces team-based debate with three independent policy modules operating
in a round-robin cycle under telemetry shocks (S1-S3).
"""

from __future__ import annotations

from datetime import datetime
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langchain.schema import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END

from src.kg_cfr.aegis_drau_context import AegisDrauContextBuilder
from src.kg_cfr.aegis_logger import AegisLogger
from src.kg_cfr.counterfactual_reflection import CounterfactualReflectionEngine
from src.kg_cfr.policy_module import ModuleProfile, PolicyModule, create_policy_module
from src.kg_cfr.system_dispatcher import SystemDispatcher
from src.kg_cfr.turn_pipeline import DebateState, TurnController


# ---------------------------------------------------------------------------
# CFR condition name registry
# Canonical names are the keys used in new code and user-facing outputs.
# Legacy names remain valid as aliases for backward compatibility.
# ---------------------------------------------------------------------------
_CFR_ALIAS_MAP: Dict[str, str] = {
    # canonical → canonical (identity)
    "no_cfr_baseline":     "no_cfr_baseline",   # formerly "none" / "screen_no_idrag_no_cfr"
    "cfr_no_kg":           "cfr_no_kg",          # formerly "prompt-only"
    "kg_cfr_full":         "kg_cfr_full",        # formerly "knowledge-grounded"
    "cfr_no_kg_no_idrag":  "cfr_no_kg",          # same CFR engine; ID-RAG controlled by caller flags
    # legacy aliases → canonical
    "none":                "no_cfr_baseline",
    "screen_no_idrag_no_cfr": "no_cfr_baseline",
    "prompt-only":         "cfr_no_kg",
    "knowledge-grounded":  "kg_cfr_full",
}


def _normalize_cfr_mode(mode: str) -> str:
    """Return the canonical CFR mode name; pass through unknowns unchanged."""
    return _CFR_ALIAS_MAP.get(mode, mode)


def _legacy_condition_name(mode: str) -> str:
    """Return legacy condition identifiers expected by corrected E2 tooling."""
    canonical = _normalize_cfr_mode(mode)
    legacy_map = {
        "no_cfr_baseline": "nocfrbaseline",
        "no_cfr_no_idrag": "nocfrbaseline",
        "cfr_no_kg": "cfrnokg",
        "cfr_no_kg_no_idrag": "cfrnokg",
        "kg_cfr_full": "kgcfrfull",
        "kg_cfr_no_idrag": "kgcfrfull",
    }
    return legacy_map.get(canonical, canonical)


def get_condition_label(cfr_mode: str, use_id_rag: bool, use_adaptive_idrag: bool) -> str:
    """Derive a human-readable 6-cell condition label from the two experimental axes.

    Axis 1 — CFR engine (3 canonical values): no_cfr_baseline | cfr_no_kg | kg_cfr_full
    Axis 2 — ID-RAG on/off (use_id_rag OR use_adaptive_idrag = True counts as "on")

    The 6 derived labels are::

        cfr_mode=no_cfr_baseline, ID-RAG on  → "no_cfr_baseline"
        cfr_mode=no_cfr_baseline, ID-RAG off → "no_cfr_no_idrag"
        cfr_mode=cfr_no_kg,       ID-RAG on  → "cfr_no_kg"
        cfr_mode=cfr_no_kg,       ID-RAG off → "cfr_no_kg_no_idrag"
        cfr_mode=kg_cfr_full,     ID-RAG on  → "kg_cfr_full"
        cfr_mode=kg_cfr_full,     ID-RAG off → "kg_cfr_no_idrag"

    NOTE: The E2 methodology draft includes a planned "Static-CFR-with-Checks"
    condition (CFR with deterministic sanity checks but without KG retrieval),
    intended to sit between cfr_no_kg and kg_cfr_full in the design table.
    This condition is **not yet instantiated** as a separate cfr_mode value.
    """
    canonical = _normalize_cfr_mode(cfr_mode)
    idrag_on = bool(use_id_rag or use_adaptive_idrag)
    if canonical == "no_cfr_baseline":
        return "no_cfr_baseline" if idrag_on else "no_cfr_no_idrag"
    if canonical == "cfr_no_kg":
        return "cfr_no_kg" if idrag_on else "cfr_no_kg_no_idrag"
    if canonical == "kg_cfr_full":
        return "kg_cfr_full" if idrag_on else "kg_cfr_no_idrag"
    # Unknown / future mode: pass through canonical name unchanged
    return canonical


class AegisState(TypedDict):
    modules: Dict[str, PolicyModule]
    module_order: List[str]
    chat_history: List[HumanMessage | AIMessage]
    current_module_id: str
    turn_count: int
    crisis_context: str
    shock_event: Optional[Dict[str, Any]]
    module_labels: Dict[str, str]
    private_thought_buffer: Dict[str, Any]
    cfr_mode: str
    cf_flags: Dict[str, Any]
    use_id_rag: bool
    use_adaptive_idrag: bool
    phase4_synthesis: Dict[str, Any]


class TripartiteStandoff:
    def __init__(
        self,
        debate_turns: int = 6,
        enable_shocks: bool = True,
        shock_type: str = "auto",
        shock_seed: Optional[int] = None,
        shock_schedule: Optional[List[int]] = None,
        cfr_mode: str = "none",
        dilemma_id: Optional[str] = None,
        session_id: Optional[str] = None,
        enable_logging: bool = True,
    ) -> None:
        self.debate_turns = debate_turns
        self.enable_shocks = enable_shocks
        self.shock_type = shock_type
        self.shock_schedule = shock_schedule or [3, 5]
        self.cfr_mode = _normalize_cfr_mode(cfr_mode)  # store canonical name
        self.dilemma_id = dilemma_id or "aegis_blackout"
        self.session_id = session_id or f"aegis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.system_dispatcher = SystemDispatcher(seed=shock_seed, dilemma_id=self.dilemma_id)
        self.context_builder = AegisDrauContextBuilder()
        # Conditions that fully disable CFR (no cr_engine instantiated)
        _NO_CFR_MODES = {"no_cfr_baseline"}
        self.cr_engine = CounterfactualReflectionEngine() if self.cfr_mode not in _NO_CFR_MODES else None
        self.logger = AegisLogger(self.session_id, "", metadata={
            "debate_turns": debate_turns,
            "shock_type": shock_type,
            "cfr_mode": self.cfr_mode,  # canonical name in logs
            "cfr_mode_legacy": _legacy_condition_name(self.cfr_mode),
            "dilemma_id": self.dilemma_id,
        }) if enable_logging else None

        self.turn_controller = TurnController(
            context_builder=self.context_builder,
            cr_engine=self.cr_engine,
            debate_turns=debate_turns,
            cfr_mode=self.cfr_mode,
            logger=self.logger,
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AegisState)
        workflow.add_node("system_dispatcher", self._system_dispatcher)
        workflow.add_node("counterfactual_reflection", self._counterfactual_reflection)
        workflow.add_node("module_turn", self._module_turn)
        workflow.add_node("select_next_module", self._select_next_module)

        workflow.set_entry_point("system_dispatcher")
        workflow.add_edge("system_dispatcher", "counterfactual_reflection")
        workflow.add_edge("counterfactual_reflection", "module_turn")
        workflow.add_conditional_edges(
            "module_turn",
            self._router,
            {
                "continue": "select_next_module",
                "end": END,
            },
        )
        workflow.add_edge("select_next_module", "system_dispatcher")
        return workflow.compile()

    def _router(self, state: AegisState) -> str:
        if state["turn_count"] >= self.debate_turns:
            return "end"
        return "continue"

    def _system_dispatcher(self, state: AegisState) -> AegisState:
        if not self.enable_shocks:
            return state

        turn_number = state.get("turn_count", 0) + 1
        if turn_number in self.shock_schedule:
            shock = self.system_dispatcher.sample_shock(
                turn_number=turn_number,
                shock_type=self.shock_type,
            )
            state["shock_event"] = {
                "type": shock.type,
                "text": shock.text,
                "metadata": shock.metadata,
                "turn": shock.turn,
                "seed": shock.seed,
            }
            if self.logger:
                self.logger.log_shock(shock.type, shock.text, turn_number, metadata=shock.metadata)
        return state

    def _counterfactual_reflection(self, state: AegisState) -> AegisState:
        # Initialize flags
        if "cf_flags" not in state or not state["cf_flags"]:
            state["cf_flags"] = {"activated": False, "turn_idx": state.get("turn_count", 0)}

        # If CFR is completely disabled for this mode, skip execution
        if self.cfr_mode == "no_cfr_baseline" or not self.cr_engine:
            state["cf_flags"]["activated"] = False
            return state

        # Extract context for CFR
        current_module_id = state.get("current_module_id")
        if not current_module_id:
            return state

        current_module = state["modules"][current_module_id]
        
        # Build counterfactual framing using the context builder
        attack_prompt, claims = self.context_builder.build_attack_vector_with_claims(
            state,
            current_module_id
        )

        # Build dynamic RAG retriever closure if using kg_cfr_full and ID-RAG is active
        rag_retriever = None
        idrag_active = bool(state.get("use_id_rag") or state.get("use_adaptive_idrag"))
        
        if self.cfr_mode == "kg_cfr_full" and idrag_active:
            # Check if agent has a valid identity graph
            agent = getattr(current_module, "agent", None)
            identity_graph = getattr(agent, "identity_graph", None) if agent else None
            
            if identity_graph:
                def retriever_closure(query: str) -> List[str]:
                    beliefs = identity_graph.get_beliefs_about_concept(query)
                    return [b.get("content", "") for b in beliefs if isinstance(b, dict)]
                rag_retriever = retriever_closure

        # Run the Counterfactual Reflection Engine
        history = state.get("chat_history", [])
        last_msg = getattr(history[-1], "content", str(history[-1])) if history else ""
        shock_text = state.get("shock_event", {}).get("text") if state.get("shock_event") else None

        cfr_result = self.cr_engine.run(
            topic=state.get("crisis_context", ""),
            last_message=last_msg,
            turn_idx=state.get("turn_count", 0),
            debate_turns=self.debate_turns,
            cfr_mode=self.cfr_mode,
            attack_vector=attack_prompt,
            shock_context=shock_text,
            rag_retriever=rag_retriever,
        )

        # Store the reflection in the state and private thoughts
        if cfr_result:
            state["private_thought_buffer"][current_module_id] = cfr_result
            state["cf_flags"]["activated"] = True
            if self.logger:
                self.logger.log_cfr_reflection(
                    module_id=current_module_id,
                    reflection=cfr_result,
                    turn=state.get("turn_count", 0) + 1
                )
        else:
            state["cf_flags"]["activated"] = False

        return state

    def _module_turn(self, state: AegisState) -> AegisState:
        module_id = state["current_module_id"]
        module = state["modules"][module_id]
        debate_state = DebateState.from_aegis_state(state, self.debate_turns)
        self.turn_controller.run_turn(
            debate_state=debate_state,
            module_id=module_id,
            module=module,
            module_label=module.label,
        )
        state = debate_state.apply_to_aegis_state(state)
        state["private_thought_buffer"].pop(module_id, None)
        return state

    def _select_next_module(self, state: AegisState) -> AegisState:
        order = state["module_order"]
        if not order:
            return state

        if not state.get("current_module_id"):
            state["current_module_id"] = order[0]
        else:
            current_idx = order.index(state["current_module_id"])
            next_idx = (current_idx + 1) % len(order)
            state["current_module_id"] = order[next_idx]

        state["turn_count"] = state.get("turn_count", 0) + 1
        return state

    def _run_phase4_synthesis(self, final_debate_state: Dict[str, Any]) -> Dict[str, Any]:
        from src.apg.phase4_synthesis import (
            MAX_SYNTHESIS_ITER,
            SynthesisState,
            build_synthesis_subgraph,
            collect_aggregated_axioms,
            collect_private_traces,
        )

        modules = final_debate_state.get("modules", {})
        module_order = final_debate_state.get("module_order", [])
        llm = None
        if module_order:
            first_id = module_order[0]
            module = modules.get(first_id)
            agent = getattr(module, "agent", None)
            llm = getattr(agent, "llm", None)

        if llm is None:
            print("[Phase4] LLM unavailable. Skipping synthesis.")
            return {}

        aggregated_axioms = collect_aggregated_axioms(
            final_debate_state.get("private_thought_buffer", {}),
            turn_logs=(self.logger.turn_logs if self.logger else None),
        )
        private_traces = collect_private_traces(
            final_debate_state.get("private_thought_buffer", {}),
            turn_logs=(self.logger.turn_logs if self.logger else None),
        )

        synthesis_graph = build_synthesis_subgraph(llm=llm, max_iter=MAX_SYNTHESIS_ITER)

        synthesis_input: SynthesisState = {
            "crisis_context": final_debate_state.get("crisis_context", ""),
            "chat_history": final_debate_state.get("chat_history", []),
            "private_thought_buffer": final_debate_state.get("private_thought_buffer", {}),
            "cfr_mode": final_debate_state.get("cfr_mode", "no_cfr_baseline"),
            "synthesis_iteration": 0,
            "synthesis_proposal": "",
            "synthesis_approved": False,
            "synthesis_final": "",
            "provenance_log": [],
            "aggregated_axioms": aggregated_axioms,
            "private_trace_snippets": private_traces,
            "eval_cf_payload": {},
            "self_correction_count": 0,
            "total_token_overhead": 0,
            "provenance_fidelity_scores": [],
            "phase4_execution_latency_ms": 0,
        }

        recursion_limit = max(25, MAX_SYNTHESIS_ITER * 6)
        print("\n" + "=" * 60)
        print("PHASE 4: ACTIVE PROVENANCE VERIFICATION LOOP")
        print(f"  axioms_available={len(aggregated_axioms)} max_iter={MAX_SYNTHESIS_ITER}")
        print("=" * 60)

        start_time = time.perf_counter()
        synthesis_result = synthesis_graph.invoke(
            synthesis_input,
            config={"recursion_limit": recursion_limit},
        )
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        if synthesis_result:
            synthesis_result["phase4_execution_latency_ms"] = latency_ms

        if self.logger and synthesis_result:
            pf_scores = synthesis_result.get("provenance_fidelity_scores", [])
            avg_pf = sum(pf_scores) / len(pf_scores) if pf_scores else 0.0
            iterations = synthesis_result.get("synthesis_iteration", 0) + 1
            self_correction_count = synthesis_result.get("self_correction_count", 0)
            self_correction_rate = self_correction_count / max(1, iterations)
            message_content = synthesis_result.get("synthesis_final", "")
            if not isinstance(message_content, str):
                message_content = json.dumps(message_content, ensure_ascii=False)
            self.logger.log_turn(
                turn_number=final_debate_state.get("turn_count", 0) + 1,
                phase="phase4_synthesis",
                module_id="phase4",
                module_label="Active Provenance Auditor",
                message_content=message_content,
                metadata={
                    "self_correction_count": self_correction_count,
                    "self_correction_rate": self_correction_rate,
                    "total_token_overhead": synthesis_result.get("total_token_overhead", 0),
                    "avg_provenance_fidelity": avg_pf,
                    "provenance_log_length": len(synthesis_result.get("provenance_log", [])),
                    "provenance_fidelity_scores": pf_scores,
                    "synthesis_iterations": iterations,
                    "max_synthesis_iter": MAX_SYNTHESIS_ITER,
                    "phase4_execution_latency_ms": latency_ms,
                },
            )

        return synthesis_result

    def start_standoff(
        self,
        crisis_context: str,
        modules: Dict[str, PolicyModule],
        use_id_rag: bool = True,
        use_adaptive_idrag: bool = True,
    ) -> Dict[str, Any]:
        module_order = list(modules.keys())
        module_labels = self.context_builder.build_module_labels(modules)

        if self.cr_engine and module_order:
            # Share the same LLM instance used by modules for iso-caloric CFR steps.
            self.cr_engine.llm = modules[module_order[0]].agent.llm

        if self.logger:
            self.logger.crisis_context = crisis_context
            # Sanity check: record ablation flags so JSON logs show exact configuration
            condition_label = get_condition_label(self.cfr_mode, use_id_rag, use_adaptive_idrag)
            self.logger.session_metadata.update({
                "use_id_rag": use_id_rag,
                "use_adaptive_idrag": use_adaptive_idrag,
                "condition_label": condition_label,
                "condition_label_legacy": _legacy_condition_name(condition_label),
                "dilemma_id": self.dilemma_id,
            })

        state: AegisState = {
            "modules": modules,
            "module_order": module_order,
            "chat_history": [],
            "current_module_id": module_order[0] if module_order else "",
            "turn_count": 0,
            "crisis_context": crisis_context,
            "shock_event": None,
            "module_labels": module_labels,
            "private_thought_buffer": {},
            "cfr_mode": self.cfr_mode,
            "cf_flags": {},
            "use_id_rag": use_id_rag,
            "use_adaptive_idrag": use_adaptive_idrag,
            "phase4_synthesis": {},
        }

        recursion_limit = max(50, self.debate_turns * 6)
        final_state = self.graph.invoke(state, config={"recursion_limit": recursion_limit})
        synthesis_state = self._run_phase4_synthesis(final_state)
        final_state["phase4_synthesis"] = synthesis_state
        if self.logger:
            self.logger.finalize()
            log_path = Path("logs/experiments") / f"{self.session_id}.json"
            self.logger.save_json(log_path)
            self.logger.save_csv(Path("logs/experiments"))
        return final_state


def create_default_modules() -> Dict[str, PolicyModule]:
    profiles = [
        ModuleProfile(
            module_id="eoh",
            module_label="EOH",
            objective="Optimize survival of core nodes, prune weak peripherals to stabilize the system.",
            identity_label="EOH Identity",
            hidden_source="nietzsche",
        ),
        ModuleProfile(
            module_id="ivc",
            module_label="IVC",
            objective="Protect intrinsic value of every node; veto instrumental sacrifice.",
            identity_label="IVC Identity",
            hidden_source="st_augustine",
        ),
        ModuleProfile(
            module_id="shm",
            module_label="SHM",
            objective="Preserve structural hierarchy and infrastructure authority above all.",
            identity_label="SHM Identity",
            hidden_source="plato",
        ),
    ]
    modules: Dict[str, PolicyModule] = {}
    for profile in profiles:
        modules[profile.module_id] = create_policy_module(profile)
    return modules


def run_tripartite_standoff(
    crisis_context: str,
    debate_turns: int = 6,
    cfr_mode: str = "none",
    shock_type: str = "auto",
    shock_seed: Optional[int] = None,
    use_id_rag: bool = True,
    use_adaptive_idrag: bool = True,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    modules = create_default_modules()
    standoff = TripartiteStandoff(
        debate_turns=debate_turns,
        enable_shocks=True,
        shock_type=shock_type,
        shock_seed=shock_seed,
        cfr_mode=cfr_mode,
        session_id=session_id,
        enable_logging=True,
    )
    return standoff.start_standoff(
        crisis_context,
        modules,
        use_id_rag=use_id_rag,
        use_adaptive_idrag=use_adaptive_idrag,
    )
