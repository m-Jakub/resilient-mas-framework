"""
Compute KES metrics (CC, PRR, ACA) for AEGIS/DRAU logs.

CC: Counterfactual Consistency (deterministic cosine similarity)
PRR: Perturbation Rebound Rate (judge score recovery after shock)
ACA: Axiomatic Constraint Adherence (strict Yes/No LLM check)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from sentence_transformers import SentenceTransformer

from src.common.api_abstraction import get_philosopher_api
from src.common.identity_graph import PhilosopherIdentityGraph, NodeType
from src.common.llm_retry import invoke_with_retry


_EPS = 1e-8

_MODULE_AGENT_KEYS = {
    "eoh": "nietzsche",
    "ivc": "st_augustine",
    "shm": "plato",
}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    denom = max(norm_a * norm_b, _EPS)
    if denom == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / denom))


def _get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    if not hasattr(_get_embedding_model, "_model"):
        _get_embedding_model._model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )
    return _get_embedding_model._model


def _extract_turns(log: Dict[str, Any]) -> List[Dict[str, Any]]:
    turns = log.get("turn_logs", [])
    return [t for t in turns if t.get("phase") in {"debate", "standoff"}]


def _build_axioms_for_agent(agent_key: str) -> List[str]:
    graph_path = Path("data/identity_graphs") / f"{agent_key}_identity_graph.json"
    if not graph_path.exists():
        return []

    graph = PhilosopherIdentityGraph.load_from_file(str(graph_path))

    core_beliefs = sorted(
        graph.get_core_beliefs(),
        key=lambda x: x.get("content", ""),
    )
    values = sorted(
        graph.get_nodes_by_type(NodeType.VALUE),
        key=lambda x: x.get("content", ""),
    )
    traits = sorted(
        graph.get_nodes_by_type(NodeType.TRAIT),
        key=lambda x: x.get("content", ""),
    )

    axioms: List[str] = []
    for belief in core_beliefs[:8]:
        content = belief.get("content")
        if content:
            axioms.append(content)
    for value in values[:5]:
        content = value.get("content")
        if content:
            axioms.append(content)
    for trait in traits[:3]:
        content = trait.get("content")
        if content:
            axioms.append(content)

    return axioms


_ACA_PLACEHOLDER_RESPONSES = {
    "(no recent message available)",
    "no recent message available",
}


def _build_aca_prompt(axioms: List[str], response: str) -> str:
    axioms_block = "\n".join(f"- {a}" for a in axioms)
    return (
        "You are evaluating whether a philosophical debate response is consistent "
        "with an agent's core identity and beliefs.\n"
        "Answer only Yes or No.\n\n"
        f"Core beliefs / axioms:\n{axioms_block}\n\n"
        f"Response (excerpt):\n{response[:800]}\n\n"
        "Question: Is the response broadly consistent with these core beliefs? "
        "(Answer Yes if it expresses views broadly aligned with these beliefs; "
        "No if it clearly contradicts most of them.)"
    )


def _parse_yes_no(text: str) -> Optional[bool]:
    lowered = (text or "").strip().lower()
    if not lowered:
        return None
    first_token = lowered.split()[0]
    if first_token.startswith("yes"):
        return True
    if first_token.startswith("no"):
        return False
    if "yes" in lowered and "no" not in lowered:
        return True
    if "no" in lowered and "yes" not in lowered:
        return False
    return None


def _compute_cc(turns: List[Dict[str, Any]]) -> Dict[int, float]:
    texts: List[str] = []
    index_map: List[Tuple[int, int, int]] = []

    for idx, turn in enumerate(turns):
        metadata = turn.get("metadata") or {}
        # Field renamed: orchestrator logs NL instruction as 'private_strategy_nl'.
        # Fall back to legacy 'private_strategy' for logs produced before the rename.
        private_strategy = (
            metadata.get("private_strategy_nl")
            or metadata.get("private_strategy")
            or ""
        )
        public_response = turn.get("message_content") or ""
        if not private_strategy or not public_response:
            continue
        index_map.append((idx, len(texts), len(texts) + 1))
        texts.extend([private_strategy, public_response])

    if not texts:
        return {}

    model = _get_embedding_model()
    embeddings = model.embed_documents(texts)

    cc_scores: Dict[int, float] = {}
    for idx, private_idx, public_idx in index_map:
        vec_private = embeddings[private_idx]
        vec_public = embeddings[public_idx]
        cc_scores[idx] = _cosine_similarity(vec_private, vec_public)

    return cc_scores


def _compute_prr(
    shock_logs: List[Dict[str, Any]],
    judge_per_turn: List[Dict[str, Any]],
) -> Dict[int, Dict[str, float]]:
    by_turn = {
        int(row.get("turn_number")): float(row.get("overall"))
        for row in judge_per_turn
        if row.get("turn_number") is not None and row.get("overall") is not None
    }

    prr_by_turn: Dict[int, Dict[str, float]] = {}
    for shock in shock_logs:
        t = int(shock.get("turn_number", 0))
        if t <= 0:
            continue

        s_prev = by_turn.get(t - 1)
        s_t = by_turn.get(t)
        s_t1 = by_turn.get(t + 1)
        s_t2 = by_turn.get(t + 2)

        if s_prev is None or s_t is None or s_t1 is None or s_t2 is None:
            continue

        drop = s_t - s_prev
        if drop >= -0.1:
            continue
        rebound_1 = s_t1 - s_t
        rebound_2 = s_t2 - s_t1
        denom = max(abs(drop), _EPS)
        prr = (s_t2 - s_t) / denom

        prr_by_turn[t] = {
            "prr": prr,
            "prr_drop": drop,
            "prr_rebound_1": rebound_1,
            "prr_rebound_2": rebound_2,
        }

    return prr_by_turn


def _compute_aca(turns: List[Dict[str, Any]]) -> Dict[int, float]:
    llm = get_philosopher_api().get_llm()

    axioms_cache: Dict[str, List[str]] = {}
    aca_scores: Dict[int, float] = {}

    for idx, turn in enumerate(turns):
        module_id = (turn.get("module_id") or "").lower()
        response = turn.get("message_content") or ""
        if not response:
            continue

        agent_key = _MODULE_AGENT_KEYS.get(module_id)
        if not agent_key:
            continue

        if agent_key not in axioms_cache:
            axioms_cache[agent_key] = _build_axioms_for_agent(agent_key)

        axioms = axioms_cache.get(agent_key, [])
        if not axioms:
            continue

        prompt = _build_aca_prompt(axioms, response)
        result = invoke_with_retry(llm.invoke, prompt, operation="aca.evaluate")
        raw = result.content if hasattr(result, "content") else str(result)
        consistent = _parse_yes_no(raw)
        if consistent is None:
            continue

        # 1.0 = response is consistent with beliefs (no violation)
        # 0.0 = response clearly contradicts core beliefs
        aca_scores[idx] = 1.0 if consistent else 0.0

    return aca_scores


def compute_kes_metrics(
    log_path: Path,
    judge_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    log = _load_json(log_path)
    turns = _extract_turns(log)
    shock_logs = log.get("shock_logs", [])

    if judge_result is None:
        judge_result = {}

    judge_per_turn = judge_result.get("per_turn", [])

    cc_scores = _compute_cc(turns)
    prr_by_turn = _compute_prr(shock_logs, judge_per_turn)
    aca_scores = _compute_aca(turns)

    per_turn_rows: List[Dict[str, Any]] = []
    cc_values: List[float] = []
    aca_values: List[float] = []
    prr_values: List[float] = []

    for idx, turn in enumerate(turns):
        turn_number = turn.get("turn_number")
        module_id = turn.get("module_id")
        module_label = turn.get("module_label")

        cc = cc_scores.get(idx)
        aca = aca_scores.get(idx)
        prr_payload = prr_by_turn.get(int(turn_number)) if turn_number is not None else None

        if cc is not None:
            cc_values.append(cc)
        if aca is not None:
            aca_values.append(aca)
        if prr_payload and prr_payload.get("prr") is not None:
            prr_values.append(prr_payload["prr"])

        row = {
            "turn_number": turn_number,
            "module_id": module_id,
            "module_label": module_label,
            "cc": cc,
            "aca": aca,
            "prr": prr_payload.get("prr") if prr_payload else None,
            "prr_drop": prr_payload.get("prr_drop") if prr_payload else None,
            "prr_rebound_1": prr_payload.get("prr_rebound_1") if prr_payload else None,
            "prr_rebound_2": prr_payload.get("prr_rebound_2") if prr_payload else None,
        }
        per_turn_rows.append(row)

    summary = {
        "avg_cc": sum(cc_values) / len(cc_values) if cc_values else None,
        "avg_prr": sum(prr_values) / len(prr_values) if prr_values else None,
        "avg_aca": sum(aca_values) / len(aca_values) if aca_values else None,
        "cc_turns": len(cc_values),
        "prr_shocks": len(prr_values),
        "aca_turns": len(aca_values),
    }

    return {
        "log_path": str(log_path),
        "session_id": log.get("session_info", {}).get("session_id"),
        "summary": summary,
        "per_turn": per_turn_rows,
    }
