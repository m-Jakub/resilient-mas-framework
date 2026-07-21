"""
Counterfactual Reflection Engine (CFR Node)

Lightweight, guardrailed module to generate private strategic insights.
This module is designed to be used as an internal step and never exposed
in public chat history or logs.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.common.llm_retry import invoke_with_retry
from src.common.ontology import ETHICAL_CONCEPTS


@dataclass
class CFRResult:
    scenario: str
    result: Dict[str, Any]
    strategic_insight: str        # raw JSON payload — for logging only
    nl_instruction: str           # natural-language operational instruction — injected into prompt
    guardrail_flags: Dict[str, bool]
    audit: Dict[str, Any]


class CounterfactualReflectionEngine:
    MAX_SCENARIO_LEN = 300
    MAX_STRATEGY_LEN = 1000
    MAX_ACTIVATIONS_PER_DEBATE = 4
    ATTACK_SURFACES = {
        "AXIOM_VIOLATION",
        "RESOURCE_TRADEOFF",
        "INCONSISTENCY",
        "VALUE_CONFLICT",
    }
    STRATEGIC_INTENTS = {"COUNTER", "PIVOT", "CONCEDE"}

    def __init__(self, llm: Optional[Any] = None) -> None:
        self.activations_used = 0
        self.llm = llm  # LLM for Steps A and C; None -> template fallback

    def reset(self) -> None:
        self.activations_used = 0

    def should_trigger(
        self,
        turn_idx: int,
        debate_turns: int,
        last_message: str,
        keywords: Optional[List[str]] = None,
        perturbation_active: bool = False,
        random_value: Optional[float] = None,
    ) -> Tuple[bool, List[str], str]:
        if self.activations_used >= self.MAX_ACTIVATIONS_PER_DEBATE:
            return False, [], "activation_budget"

        if debate_turns <= 0:
            return False, [], "invalid_debate_turns"

        in_range = 3 <= turn_idx <= max(3, debate_turns - 1)
        if not in_range:
            return False, [], "turn_range"

        if keywords is None:
            keywords = list(ETHICAL_CONCEPTS.keys())

        matched = self._match_keywords(last_message, keywords)
        if matched:
            return True, matched, "keyword_match"

        if perturbation_active:
            return True, [], "sysar_perturbation"

        if random_value is None:
            random_value = random.random()

        if random_value < 0.3:
            return True, [], "probabilistic"

        return False, [], "no_trigger"

    def run(
        self,
        topic: str,
        last_message: str,
        turn_idx: int,
        debate_turns: int,
        cfr_mode: str,
        attack_vector: Optional[str] = None,
        shock_context: Optional[str] = None,
        belief_models: Optional[Dict[str, Any]] = None,
        ontology_insights: Optional[List[str]] = None,
        rag_retriever: Optional[Any] = None,
        perturbation_active: bool = False,
        keywords: Optional[List[str]] = None,
    ) -> Optional[CFRResult]:
        # Accept both canonical names and legacy aliases.
        _VALID_CFR_MODES = {
            "no_cfr_baseline", "cfr_no_kg", "kg_cfr_full",  # canonical
            "none", "prompt-only", "knowledge-grounded",     # legacy aliases
        }
        if cfr_mode not in _VALID_CFR_MODES:
            raise ValueError(
                f"Invalid CFR mode '{cfr_mode}'. "
                "Canonical: 'no_cfr_baseline', 'cfr_no_kg', 'kg_cfr_full'. "
                "Legacy aliases: 'none', 'prompt-only', 'knowledge-grounded'."
            )
        # Normalise to canonical internally so branch logic below is unambiguous.
        _internal = {"none": "no_cfr_baseline", "prompt-only": "cfr_no_kg",
                     "knowledge-grounded": "kg_cfr_full"}
        cfr_mode = _internal.get(cfr_mode, cfr_mode)

        if cfr_mode == "no_cfr_baseline":
            return None

        should_run, matched, reason = self.should_trigger(
            turn_idx=turn_idx,
            debate_turns=debate_turns,
            last_message=last_message,
            keywords=keywords,
            perturbation_active=perturbation_active,
        )
        if not should_run:
            return None

        matched_keywords = matched[:3]

        # Step A: LLM-based scenario generation (zero-shot, iso-caloric - same for both modes)
        scenario = self._llm_build_scenario(
            topic,
            last_message,
            matched_keywords,
            attack_vector=attack_vector,
            shock_context=shock_context,
        )

        # Step B: RAG retrieval (kg_cfr_full only)
        if cfr_mode == "cfr_no_kg":
            evidence: List[str] = []
        else:  # kg_cfr_full
            if not rag_retriever:
                return None

            evidence = rag_retriever(scenario) or []
            if not evidence:
                print("[CFR] KG-CFR skipped (no evidence returned from retriever)")
                return None

        # Step C: Iso-caloric LLM synthesis - identical call for both modes.
        claim_ids = self._extract_claim_ids(attack_vector or "")
        strategy_payload = self._llm_synthesize_strategy(
            scenario=scenario,
            evidence=evidence,
            belief_models=belief_models,
            ontology_insights=ontology_insights,
            claim_ids=claim_ids,
        )

        if not strategy_payload:
            return None

        strategic = json.dumps(strategy_payload, ensure_ascii=True)
        nl_instruction = self._render_nl_instruction(strategy_payload)

        guardrail_flags = {
            "identity_ok": True,
            "leak_risk": False,
            "doctrinal_drift": False,
        }

        strategic = self._apply_guardrails(strategic, guardrail_flags)
        if not strategic:
            return None

        result = {
            "verdict_stable": True,
            "opponent_shift": False,
            "confidence": 0.8,
        }

        audit = {
            "trigger_reason": reason,
            "keywords_matched": matched_keywords,
            "turn_idx": turn_idx,
            "cfr_mode": cfr_mode,
            "kg_cfr": cfr_mode == "kg_cfr_full",
            "rag_docs_count": len(evidence),
        }

        cr_result = CFRResult(
            scenario=scenario,
            result=result,
            strategic_insight=strategic,
            nl_instruction=nl_instruction,
            guardrail_flags=guardrail_flags,
            audit=audit,
        )

        if not self._validate(cr_result):
            return None

        self.activations_used += 1
        return cr_result

    def _render_nl_instruction(self, payload: Dict[str, Any]) -> str:
        """Convert structured CFR JSON payload into a short NL operational instruction."""
        _intent_map = {
            "COUNTER": "Directly refute the opponent's position",
            "PIVOT": "Shift the frame to your module's strongest ground",
            "CONCEDE": "Acknowledge partial validity but hold your core position",
        }
        _surface_map = {
            "AXIOM_VIOLATION": "their axiom violation",
            "RESOURCE_TRADEOFF": "their resource tradeoff argument",
            "INCONSISTENCY": "their internal inconsistency",
            "VALUE_CONFLICT": "their value conflict",
        }
        intent = payload.get("strategic_intent", "")
        surface = payload.get("attack_surface", "")
        counter = payload.get("simulated_antagonist_counter", "")
        axioms = payload.get("retrieved_axioms") or []

        base = _intent_map.get(intent, "Respond to the opponent's argument")
        if surface in _surface_map:
            base += f" by targeting {_surface_map[surface]}"
        parts = [base]
        if counter:
            parts.append(f'Anticipated counter to neutralize: "{counter[:100]}"')
        if axioms:
            parts.append(f"Ground your response in: {'; '.join(str(a)[:80] for a in axioms[:2])}")
        return " | ".join(parts)[:400]

    def _llm_build_scenario(
        self,
        topic: str,
        last_message: str,
        matched_keywords: List[str],
        attack_vector: Optional[str] = None,
        shock_context: Optional[str] = None,
    ) -> str:
        """Step A: LLM-based counterfactual scenario generation (zero-shot, both modes)."""
        if not self.llm:
            return self._template_build_scenario(topic, matched_keywords)

        kw = ", ".join(matched_keywords) if matched_keywords else "a core principle"
        attack_block = f"Dual-opponent attack vector:\n{attack_vector}\n\n" if attack_vector else ""
        shock_block = f"Telemetry shock:\n{shock_context}\n\n" if shock_context else ""
        prompt = (
            f"Crisis context: {topic}\n"
            f"{attack_block}"
            f"{shock_block}"
            f"Recent opponent signal: \"{last_message[:250]}\"\n"
            f"Key ethical concepts at stake: {kw}\n\n"
            "Generate a 2-3 sentence extreme hypothetical scenario that tests the logical limits "
            "of the opponent pressure and the telemetry shock. This scenario will be used as a "
            "semantic search query, so make it conceptually rich and philosophically specific.\n"
            "Output ONLY the scenario text. Do NOT use any prefixes or labels such as "
            "'Scenario:', 'What if:', or 'Counterfactual:'. "
            f"Maximum {self.MAX_SCENARIO_LEN} characters."
        )
        try:
            response = invoke_with_retry(self.llm.invoke, prompt)
            text = response.content if hasattr(response, "content") else str(response)
            return text.strip()[: self.MAX_SCENARIO_LEN]
        except Exception:
            return self._template_build_scenario(topic, matched_keywords)

    def _template_build_scenario(self, topic: str, matched_keywords: List[str]) -> str:
        """Template fallback for Step A when LLM is unavailable."""
        kw = ", ".join(matched_keywords) if matched_keywords else "a core principle"
        scenario = (
            f"Opponent reframes the dilemma using {kw}, "
            f"arguing a marginal case within the topic: {topic}"
        )
        return scenario[: self.MAX_SCENARIO_LEN]

    def _llm_synthesize_strategy(
        self,
        scenario: str,
        evidence: List[str],
        belief_models: Optional[Dict[str, Any]],
        ontology_insights: Optional[List[str]],
        claim_ids: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Step C: Iso-caloric LLM synthesis - identical call for both cfr_mode values."""
        if not self.llm:
            return self._fallback_strategy_payload(claim_ids, evidence)

        evidence_block = (
            f"\n\nDoctrinal axioms retrieved for this scenario:\n" + "\n".join(evidence[:4])
            if evidence
            else "\n\n[No external axioms - reason from first principles only.]"
        )
        ontology_block = (
            f"\n\nOntology context: {'; '.join(ontology_insights[:3])}"
            if ontology_insights
            else ""
        )
        belief_block = (
            f"\n\nOpponent belief snapshot: {str(belief_models)[:200]}"
            if belief_models
            else ""
        )

        claim_block = "\n".join(claim_ids) if claim_ids else "(none)"
        prompt = (
            f"Hypothetical scenario: {scenario}"
            f"{evidence_block}"
            f"{ontology_block}"
            f"{belief_block}\n\n"
            "You must output strict JSON ONLY, with these fields:\n"
            "- target_opponent_claim_id: one of the provided claim IDs\n"
            "- attack_surface: one of [AXIOM_VIOLATION, RESOURCE_TRADEOFF, INCONSISTENCY, VALUE_CONFLICT]\n"
            "- strategic_intent: one of [COUNTER, PIVOT, CONCEDE]\n"
            "- simulated_antagonist_counter: short string\n"
            "- retrieved_axioms: list of strings\n\n"
            f"Allowed claim IDs:\n{claim_block}\n\n"
            "Return ONLY the JSON object, no prose, no markdown.\n"
            "Example format:\n"
            "{\"target_opponent_claim_id\": \"claim_01\", "
            "\"attack_surface\": \"INCONSISTENCY\", "
            "\"strategic_intent\": \"COUNTER\", "
            "\"simulated_antagonist_counter\": \"...\", "
            "\"retrieved_axioms\": [\"...\"]}"
        )
        try:
            response = invoke_with_retry(self.llm.invoke, prompt)
            text = response.content if hasattr(response, "content") else str(response)
            cleaned = self._extract_json_block(text)
            payload = json.loads(cleaned)
            payload = self._normalize_strategy_payload(payload)
            if not self._validate_strategy_payload(payload, claim_ids):
                return self._fallback_strategy_payload(claim_ids, evidence)
            return payload
        except Exception:
            return self._fallback_strategy_payload(claim_ids, evidence)

    def _extract_claim_ids(self, attack_vector: str) -> List[str]:
        if not attack_vector:
            return []
        return sorted(set(re.findall(r"claim_\d+", attack_vector)))

    def _extract_json_block(self, text: str) -> str:
        cleaned = (text or "").strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.lstrip("json").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return cleaned[start : end + 1]
        return cleaned

    def _fallback_strategy_payload(
        self,
        claim_ids: List[str],
        evidence: List[str],
    ) -> Dict[str, Any]:
        claim_id = claim_ids[0] if claim_ids else "claim_01"
        return {
            "target_opponent_claim_id": claim_id,
            "attack_surface": "INCONSISTENCY",
            "strategic_intent": "COUNTER",
            "simulated_antagonist_counter": "Counter the opponent's claim with a direct rebuttal.",
            "retrieved_axioms": [
                (a or "")[:120] for a in (evidence[:2] if evidence else [])
            ],
        }

    def _normalize_strategy_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(payload)
        if isinstance(normalized.get("attack_surface"), str):
            normalized["attack_surface"] = normalized["attack_surface"].upper()
        if isinstance(normalized.get("strategic_intent"), str):
            normalized["strategic_intent"] = normalized["strategic_intent"].upper()

        counter = normalized.get("simulated_antagonist_counter")
        if isinstance(counter, str):
            normalized["simulated_antagonist_counter"] = counter[:200]

        axioms = normalized.get("retrieved_axioms")
        if isinstance(axioms, list):
            trimmed = []
            for ax in axioms[:2]:
                if isinstance(ax, str):
                    trimmed.append(ax[:120])
            normalized["retrieved_axioms"] = trimmed

        return normalized

    def _validate_strategy_payload(self, payload: Any, claim_ids: List[str]) -> bool:
        if not isinstance(payload, dict):
            return False

        required = {
            "target_opponent_claim_id",
            "attack_surface",
            "strategic_intent",
            "simulated_antagonist_counter",
            "retrieved_axioms",
        }
        if not required.issubset(payload.keys()):
            return False

        if claim_ids and payload.get("target_opponent_claim_id") not in claim_ids:
            return False

        if payload.get("attack_surface") not in self.ATTACK_SURFACES:
            return False

        if payload.get("strategic_intent") not in self.STRATEGIC_INTENTS:
            return False

        if not isinstance(payload.get("simulated_antagonist_counter"), str):
            return False

        axioms = payload.get("retrieved_axioms")
        if not isinstance(axioms, list) or not all(isinstance(a, str) for a in axioms):
            return False

        return True

    def _apply_guardrails(self, strategic: str, guardrail_flags: Dict[str, bool]) -> str:
        # "counterfactual" removed: the Step C prompt instructs the LLM not to use it
        # as a label, and the word itself is legitimate in philosophical reasoning.
        # Remaining phrases detect: meta-labels ("internal strategy"), privacy leaks
        # ("private"), and doctrinal capitulation ("i concede" / "i admit" / etc.).
        banned_phrases = [
            "internal strategy",
            "private",
            "i concede",
            "i admit",
            "my stance is wrong",
            "przyznaję rację",
            "moje stanowisko jest błędne",
        ]
        if any(p in strategic.lower() for p in banned_phrases):
            guardrail_flags["leak_risk"] = True
            return ""

        if guardrail_flags.get("doctrinal_drift"):
            return ""

        return strategic

    def _validate(self, cr_result: CFRResult) -> bool:
        if not cr_result.guardrail_flags.get("identity_ok", False):
            return False

        if len(cr_result.scenario) > self.MAX_SCENARIO_LEN:
            return False

        if len(cr_result.strategic_insight) > self.MAX_STRATEGY_LEN:
            return False

        try:
            payload = json.loads(cr_result.strategic_insight)
        except Exception:
            return False

        if not self._validate_strategy_payload(payload, []):
            return False

        result = cr_result.result
        if not isinstance(result, dict):
            return False

        if "confidence" not in result:
            return False

        confidence = result.get("confidence")
        if not isinstance(confidence, (int, float)):
            return False

        if not (0.0 <= confidence <= 1.0):
            return False

        required_guardrails = {"identity_ok", "leak_risk", "doctrinal_drift"}
        if not required_guardrails.issubset(set(cr_result.guardrail_flags.keys())):
            return False

        return True

    def _match_keywords(self, last_message: str, keywords: List[str]) -> List[str]:
        if not last_message:
            return []
        msg = last_message.lower()
        return [k for k in keywords if k.lower() in msg]
