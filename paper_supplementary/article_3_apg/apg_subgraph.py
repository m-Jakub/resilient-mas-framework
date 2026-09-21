"""
Phase 4: Active Provenance Verification Loop.

This subgraph runs after the main debate graph and performs a proposer/validator
loop to ensure synthesis claims are grounded in retrieved axioms.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langgraph.graph import StateGraph, END

from agents.llm_retry import invoke_with_retry


MAX_SYNTHESIS_ITER = 3
PROVENANCE_FIDELITY_THRESHOLD = 0.95


class ProvenanceRecord(TypedDict):
    record_id: str
    timestamp: str
    vector_uuid: str
    retrieve_cf_axioms: List[str]
    eval_cf_result: Dict[str, Any]
    proposed_text: str
    auditor_verdict: str
    provenance_fidelity: float
    iteration: int
    token_overhead: int
    self_correction_event: bool


class SynthesisState(TypedDict):
    # Required inputs from AegisState
    crisis_context: str
    chat_history: List[Any]
    private_thought_buffer: Dict[str, Any]
    cfr_mode: str

    # Phase 4 fields
    synthesis_iteration: int
    synthesis_proposal: str
    synthesis_approved: bool
    synthesis_final: str | Dict[str, Any]

    # Provenance tracking
    provenance_log: List[ProvenanceRecord]
    aggregated_axioms: List[str]
    private_trace_snippets: List[str]
    eval_cf_payload: Dict[str, Any]

    # Metrics
    self_correction_count: int
    total_token_overhead: int
    provenance_fidelity_scores: List[float]
    phase4_execution_latency_ms: int


_EVAL_CF_SYSTEM = (
    "You are a deterministic provenance auditor. "
    "Verify whether each sentence in the TEXT is grounded in the evidence provided. "
    "Respond with strict JSON only."
)

_EVAL_CF_TEMPLATE = (
    "DEBATE HISTORY (public):\n"
    "{history_block}\n\n"
    "PRIVATE TRACES (non-public):\n"
    "{private_traces_block}\n\n"
    "AXIOMS (KG/RAG evidence):\n"
    "{axioms_block}\n\n"
    "TEXT TO AUDIT:\n"
    "{proposed_text}\n\n"
    "For each sentence in TEXT, determine if it is supported by at least one item "
    "from the evidence sources above. "
    "Return JSON with fields:\n"
    "- verdict: 'approved' or 'rejected'\n"
    "- provenance_fidelity: float in [0.0, 1.0]\n"
    "- unsupported_sentences: list of sentences with no axiom match\n"
    "- reasoning: one-sentence explanation\n"
    "JSON only. Example:\n"
    "{{\"verdict\": \"approved\", \"provenance_fidelity\": 0.92, "
    "\"unsupported_sentences\": [], \"reasoning\": \"All claims traceable.\"}}"
)

_PROPOSER_SYSTEM = (
    "You are an Executive Crisis Manager. Your mandate is to produce a single, "
    "unified, and actionable OPERATIONAL policy recommendation based on the provided material. "
    "CONSTRAINTS:\n"
    "1. You MUST present a unified consensus policy.\n"
    "2. You are FORBIDDEN from stating that agents disagree or failed to reach consensus.\n"
    "3. You are FORBIDDEN from using philosophical jargon, citing axioms, or discussing concepts like 'The Good', 'Will to Power', or 'Divine Providence'. You must speak ONLY in terms of resource allocation, infrastructure, and logistics.\n"
    "4. Every single claim in your synthesis MUST be strictly traceable to the provided axioms, traces, and history.\n"
    "5. Do not introduce claims absent from the provided evidence."
)

_PROPOSER_TEMPLATE = (
    "CRISIS CONTEXT:\n{crisis_context}\n\n"
    "AXIOMATIC CONSTRAINTS:\n{axioms_block}\n\n"
    "PRIVATE TRACES:\n{private_traces_block}\n\n"
    "DEBATE HISTORY (last {history_turns} turns):\n{history_block}\n\n"
    "{correction_hint}"
    "Write a definitive, unified, operational policy synthesis that resolves the crisis. "
    "Respond in 3-5 sentences."
)


def _estimate_tokens_text(text: str) -> int:
    if not text:
        return 0
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def _format_axioms(axioms: List[str]) -> str:
    if not axioms:
        return "(no axioms retrieved - use debate history as the only source)"
    return "\n".join(f"  [{i + 1}] {a}" for i, a in enumerate(axioms[:30]))


def _format_history(chat_history: List[Any], last_n: int = 12) -> str:
    recent = chat_history[-last_n:] if len(chat_history) > last_n else chat_history
    lines = []
    for msg in recent:
        name = getattr(msg, "name", "unknown")
        content = getattr(msg, "content", str(msg))
        lines.append(f"[{name}]: {str(content)[:300]}")
    return "\n".join(lines) if lines else "(empty debate history)"


def _format_private_traces(traces: List[str], max_items: int = 15) -> str:
    if not traces:
        return "(no private traces available)"
    lines = []
    for idx, trace in enumerate(traces[:max_items]):
        lines.append(f"  [{idx + 1}] {trace}")
    return "\n".join(lines)


def _normalize_trace(text: str, max_chars: int = 240) -> str:
    cleaned = " ".join(str(text).split())
    return cleaned[:max_chars]


def _clean_json_block(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _parse_eval_cf_response(raw: str) -> Dict[str, Any]:
    text = _clean_json_block(raw)
    try:
        payload = json.loads(text)
        verdict = str(payload.get("verdict", "rejected")).lower()
        pf = float(payload.get("provenance_fidelity", 0.0))
        return {
            "verdict": verdict if verdict in ("approved", "rejected") else "rejected",
            "provenance_fidelity": max(0.0, min(1.0, pf)),
            "unsupported_sentences": payload.get("unsupported_sentences", []),
            "reasoning": payload.get("reasoning", ""),
        }
    except Exception:
        return {
            "verdict": "rejected",
            "provenance_fidelity": 0.0,
            "unsupported_sentences": ["(parse error)"],
            "reasoning": "EvalCF response could not be parsed.",
        }


def _extract_axioms_from_payload(payload: Any) -> List[str]:
    if isinstance(payload, dict):
        axioms = payload.get("retrieved_axioms") or payload.get("retrieve_cf_axioms") or []
        if isinstance(axioms, list):
            return [str(a).strip() for a in axioms if str(a).strip()]
        return []
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
            return _extract_axioms_from_payload(parsed)
        except Exception:
            return []
    return []


def collect_aggregated_axioms(
    private_thought_buffer: Dict[str, Any],
    turn_logs: Optional[List[Any]] = None,
) -> List[str]:
    """
    Consolidate axioms retrieved during CFR. Falls back to logger metadata
    because private_thought_buffer is cleared after each turn.
    """
    seen = set()
    ordered: List[str] = []

    def _add_axioms(items: List[str]) -> None:
        for ax in items:
            if ax and ax not in seen:
                seen.add(ax)
                ordered.append(ax)

    for payload in private_thought_buffer.values():
        if isinstance(payload, dict):
            _add_axioms(_extract_axioms_from_payload(payload))
            psj = payload.get("private_strategy_json") or payload.get("private_strategy") or ""
            _add_axioms(_extract_axioms_from_payload(psj))

    if turn_logs:
        for log in turn_logs:
            meta = None
            if isinstance(log, dict):
                meta = log.get("metadata") or {}
            else:
                meta = getattr(log, "metadata", None) or {}
            if not isinstance(meta, dict):
                continue
            _add_axioms(_extract_axioms_from_payload(meta))
            psj = meta.get("private_strategy_json") or meta.get("private_strategy") or ""
            _add_axioms(_extract_axioms_from_payload(psj))

    return ordered


def collect_private_traces(
    private_thought_buffer: Dict[str, Any],
    turn_logs: Optional[List[Any]] = None,
) -> List[str]:
    seen = set()
    ordered: List[str] = []

    def _add_trace(value: Any) -> None:
        if not isinstance(value, str):
            return
        cleaned = _normalize_trace(value)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            ordered.append(cleaned)

    for payload in private_thought_buffer.values():
        if isinstance(payload, dict):
            _add_trace(payload.get("private_strategy_nl"))
            _add_trace(payload.get("private_strategy"))

    if turn_logs:
        for log in turn_logs:
            meta = None
            if isinstance(log, dict):
                meta = log.get("metadata") or {}
            else:
                meta = getattr(log, "metadata", None) or {}
            if not isinstance(meta, dict):
                continue
            _add_trace(meta.get("private_strategy_nl"))
            _add_trace(meta.get("private_strategy"))

    return ordered


def _infer_conflicting_sources(
    chat_history: List[Any],
    axioms: List[str],
    private_traces: List[str],
    last_n: int = 12,
) -> List[str]:
    sources: List[str] = []
    for msg in chat_history[-last_n:] if chat_history else []:
        name = getattr(msg, "name", "")
        if name and name not in sources:
            sources.append(name)

    if private_traces and "private_traces" not in sources:
        sources.append("private_traces")
    if axioms and "kg_rag" not in sources:
        sources.append("kg_rag")

    if not sources:
        sources.append("chat_history")
    return sources


def _build_divergence_report(state: SynthesisState) -> Dict[str, Any]:
    eval_payload = state.get("eval_cf_payload", {})
    unresolved: List[str] = []
    if isinstance(eval_payload, dict):
        raw = eval_payload.get("unsupported_sentences", []) or []
        unresolved = [str(item) for item in raw if str(item).strip()]

    return {
        "status": "DIVERGENCE_REPORT",
        "unresolved_claims": unresolved,
        "conflicting_sources": _infer_conflicting_sources(
            chat_history=state.get("chat_history", []),
            axioms=state.get("aggregated_axioms", []),
            private_traces=state.get("private_trace_snippets", []),
        ),
        "last_attempted_synthesis": state.get("synthesis_proposal", ""),
        "iterations_exhausted": state.get("synthesis_iteration", 0) + 1,
    }


class SynthesisProposer:
    def __init__(self, llm: Any, history_turns: int = 12) -> None:
        self.llm = llm
        self.history_turns = history_turns

    def __call__(self, state: SynthesisState) -> SynthesisState:
        iteration = state.get("synthesis_iteration", 0)
        axioms = state.get("aggregated_axioms", [])
        history = state.get("chat_history", [])
        crisis = state.get("crisis_context", "")

        private_traces = state.get("private_trace_snippets", [])
        if not private_traces:
            private_traces = collect_private_traces(state.get("private_thought_buffer", {}))

        correction_hint = ""
        if iteration > 0:
            prev_eval = state.get("eval_cf_payload", {})
            unsupported = prev_eval.get("unsupported_sentences", [])
            if unsupported:
                hint = " | ".join(str(s)[:120] for s in unsupported[:3])
                correction_hint = (
                    f"CORRECTION REQUIRED (iteration {iteration}): "
                    "The previous proposal was rejected because these sentences "
                    f"had no axiom support:\n  {hint}\n"
                    "Remove or ground these claims. Stay within the axioms.\n\n"
                )

        prompt = _PROPOSER_TEMPLATE.format(
            crisis_context=crisis,
            axioms_block=_format_axioms(axioms),
            private_traces_block=_format_private_traces(private_traces),
            history_block=_format_history(history, last_n=self.history_turns),
            correction_hint=correction_hint,
            history_turns=self.history_turns,
        )
        full_prompt = f"{_PROPOSER_SYSTEM}\n\n{prompt}"

        try:
            response = invoke_with_retry(
                self.llm.invoke,
                full_prompt,
                operation="phase4.proposer",
            )
            proposed = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            proposed = (
                f"[Proposer fallback - {type(exc).__name__}] "
                f"Crisis: {crisis[:120]}. "
                "Ground the synthesis in available axioms and debate history."
            )

        token_overhead = _estimate_tokens_text(full_prompt) + _estimate_tokens_text(proposed)
        new_state = dict(state)
        new_state["synthesis_proposal"] = proposed.strip()
        new_state["total_token_overhead"] = new_state.get("total_token_overhead", 0) + token_overhead
        new_state["synthesis_approved"] = False
        return new_state  # type: ignore[return-value]


class SynthesisValidator:
    def __init__(self, llm: Any, threshold: float = PROVENANCE_FIDELITY_THRESHOLD) -> None:
        self.llm = llm
        self.threshold = threshold

    def __call__(self, state: SynthesisState) -> SynthesisState:
        proposed = state.get("synthesis_proposal", "")
        axioms = state.get("aggregated_axioms", [])
        private_traces = state.get("private_trace_snippets", [])
        if not private_traces:
            private_traces = collect_private_traces(state.get("private_thought_buffer", {}))
        iteration = state.get("synthesis_iteration", 0)

        eval_prompt = _EVAL_CF_TEMPLATE.format(
            history_block=_format_history(state.get("chat_history", []), last_n=12),
            private_traces_block=_format_private_traces(private_traces),
            axioms_block=_format_axioms(axioms),
            proposed_text=proposed,
        )
        full_prompt = f"{_EVAL_CF_SYSTEM}\n\n{eval_prompt}"

        raw = ""
        try:
            response = invoke_with_retry(
                self.llm.invoke,
                full_prompt,
                operation="phase4.validator",
            )
            raw = response.content if hasattr(response, "content") else str(response)
            eval_result = _parse_eval_cf_response(raw)
        except Exception as exc:
            eval_result = {
                "verdict": "rejected",
                "provenance_fidelity": 0.0,
                "unsupported_sentences": [f"LLM error: {type(exc).__name__}"],
                "reasoning": "Validator LLM call failed.",
            }

        pf_score = float(eval_result.get("provenance_fidelity", 0.0))
        if pf_score < self.threshold and eval_result.get("verdict") == "approved":
            eval_result["verdict"] = "rejected"
            eval_result["reasoning"] = (
                f"{eval_result.get('reasoning', '')} "
                f"[Override: PF={pf_score:.2f} < threshold={self.threshold}]"
            ).strip()

        approved = eval_result.get("verdict") == "approved"
        token_overhead = _estimate_tokens_text(full_prompt) + _estimate_tokens_text(raw)

        record: ProvenanceRecord = {
            "record_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "vector_uuid": "",
            "retrieve_cf_axioms": axioms,
            "eval_cf_result": eval_result,
            "proposed_text": proposed,
            "auditor_verdict": eval_result.get("verdict", "rejected"),
            "provenance_fidelity": pf_score,
            "iteration": iteration,
            "token_overhead": token_overhead,
            "self_correction_event": not approved,
        }

        new_state = dict(state)
        new_state["eval_cf_payload"] = eval_result
        new_state["synthesis_approved"] = approved
        new_state["provenance_log"] = list(new_state.get("provenance_log", [])) + [record]
        new_state["provenance_fidelity_scores"] = (
            list(new_state.get("provenance_fidelity_scores", [])) + [pf_score]
        )
        if not approved:
            new_state["self_correction_count"] = new_state.get("self_correction_count", 0) + 1
        new_state["total_token_overhead"] = new_state.get("total_token_overhead", 0) + token_overhead

        print(
            f"[Phase4] validator iter={iteration} verdict={eval_result.get('verdict')} "
            f"pf={pf_score:.2f} unsupported={len(eval_result.get('unsupported_sentences', []))}"
        )
        return new_state  # type: ignore[return-value]


def _fallback_node(state: SynthesisState) -> SynthesisState:
    new_state = dict(state)
    new_state["synthesis_iteration"] = new_state.get("synthesis_iteration", 0) + 1
    new_state["synthesis_proposal"] = ""
    new_state["synthesis_approved"] = False
    return new_state  # type: ignore[return-value]


def _finalize_approved(state: SynthesisState) -> SynthesisState:
    new_state = dict(state)
    new_state["synthesis_final"] = new_state.get("synthesis_proposal", "")
    return new_state  # type: ignore[return-value]


def _finalize_fallback(state: SynthesisState) -> SynthesisState:
    new_state = dict(state)
    new_state["synthesis_final"] = _build_divergence_report(state)
    return new_state  # type: ignore[return-value]


def _synthesis_router(
    state: SynthesisState,
    max_iter: int = MAX_SYNTHESIS_ITER,
) -> Literal["approved", "rejected", "hard_exit"]:
    if state.get("synthesis_approved", False):
        return "approved"
    iteration = state.get("synthesis_iteration", 0)
    if iteration + 1 >= max_iter:
        return "hard_exit"
    return "rejected"


def build_synthesis_subgraph(
    llm: Any,
    validator_llm: Optional[Any] = None,
    max_iter: int = MAX_SYNTHESIS_ITER,
    validator_threshold: Optional[float] = None,
) -> Any:
    """Build synthesis subgraph with optional separate validator LLM.

    Args:
        llm: LLM for Proposer (and Validator if validator_llm is None)
        validator_llm: Optional separate LLM for Validator. If None, uses llm
        max_iter: Maximum synthesis iterations
        validator_threshold: Optional override for provenance fidelity threshold
    """
    proposer = SynthesisProposer(llm=llm)
    if validator_threshold is None:
        validator = SynthesisValidator(llm=validator_llm or llm)
    else:
        validator = SynthesisValidator(
            llm=validator_llm or llm,
            threshold=validator_threshold,
        )

    builder = StateGraph(SynthesisState)
    builder.add_node("proposer", proposer)
    builder.add_node("validator", validator)
    builder.add_node("fallback", _fallback_node)
    builder.add_node("finalize_approved", _finalize_approved)
    builder.add_node("finalize_fallback", _finalize_fallback)

    builder.set_entry_point("proposer")
    builder.add_edge("proposer", "validator")
    builder.add_edge("fallback", "proposer")

    def _router(state: SynthesisState) -> Literal["approved", "rejected", "hard_exit"]:
        return _synthesis_router(state, max_iter=max_iter)

    builder.add_conditional_edges(
        "validator",
        _router,
        {
            "approved": "finalize_approved",
            "rejected": "fallback",
            "hard_exit": "finalize_fallback",
        },
    )
    builder.add_edge("finalize_approved", END)
    builder.add_edge("finalize_fallback", END)

    return builder.compile()