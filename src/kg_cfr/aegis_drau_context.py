"""
AEGIS/DRAU context builder.

Builds module-labeled perspectives (EOH/IVC/SHM) and the Dual-Opponent
Attack Vector A_opp for CFR input.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from src.common.utils.claim_segmenter import extract_claims


class AegisDrauContextBuilder:
    """Builds AEGIS/DRAU module contexts without exposing team framing."""

    def __init__(self, module_labels: Optional[List[str]] = None) -> None:
        self.module_labels = module_labels or ["EOH", "IVC", "SHM"]

    def build_module_labels(self, modules: Dict[str, Any]) -> Dict[str, str]:
        """Assign EOH/IVC/SHM labels to teams deterministically."""
        if not modules:
            return {}

        labels = self.module_labels
        mapping: Dict[str, str] = {}

        # Prefer explicit label hints in team_id or team.name if present.
        for module_id, module in modules.items():
            lowered = f"{module_id} {getattr(module, 'label', '')}".lower()
            if "eoh" in lowered:
                mapping[module_id] = "EOH"
            elif "ivc" in lowered:
                mapping[module_id] = "IVC"
            elif "shm" in lowered:
                mapping[module_id] = "SHM"

        # Fill remaining slots by stable ordering.
        remaining = [m for m in sorted(modules.keys()) if m not in mapping]
        label_iter = [l for l in labels if l not in mapping.values()]
        for module_id, label in zip(remaining, label_iter):
            mapping[module_id] = label

        # Fallback for any leftover teams.
        for module_id in remaining[len(label_iter):]:
            mapping[module_id] = f"MODULE-{len(mapping) + 1}"

        return mapping

    def _last_messages_by_module(self, chat_history: List[Any]) -> Dict[str, str]:
        """Extract last message content per module_id from chat history."""
        last_by_module: Dict[str, str] = {}
        for msg in reversed(chat_history or []):
            name = getattr(msg, "name", "") or ""
            if ":" in name:
                module_id = name.split(":", 1)[0]
                if module_id not in last_by_module and msg.content:
                    last_by_module[module_id] = str(msg.content)
            if len(last_by_module) >= 3:
                break
        return last_by_module

    def _extract_shock_fields(self, shock_event: Any) -> tuple[str, str]:
        """Extract shock type/text from either dataclass or dict payloads."""
        if shock_event is None:
            return "unknown", ""

        if hasattr(shock_event, "type") and hasattr(shock_event, "text"):
            return str(shock_event.type), str(shock_event.text)

        if isinstance(shock_event, dict):
            return str(shock_event.get("type", "unknown")), str(shock_event.get("text", ""))

        return "unknown", str(shock_event)

    def _format_claims(self, claims: List[Dict[str, str]]) -> str:
        if not claims:
            return "(no claims extracted)"
        return "\n".join(
            f"  [{c['claim_id']}]: {c['claim_text']}" for c in claims
        )

    def build_attack_vector_with_claims(
        self,
        state: Dict[str, Any],
        current_module_id: str,
        max_chars: int = 2500,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Build A_opp plus the extracted opponent claims for logging."""
        modules = state.get("modules", {})
        module_labels = state.get("module_labels") or self.build_module_labels(modules)
        last_msgs = self._last_messages_by_module(state.get("chat_history", []))
        shock_event = state.get("shock_event") or state.get("perturbation")

        lines: List[str] = ["Dual-opponent attack vector (A_opp):"]
        competitors = [m for m in modules.keys() if m != current_module_id]

        competitors.sort(key=lambda t: module_labels.get(t, t))

        claims_payload: List[Dict[str, Any]] = []
        for comp_id in competitors[:2]:
            label = module_labels.get(comp_id, "MODULE")
            comp_msg = last_msgs.get(comp_id, "(no recent message available)")
            claims = extract_claims(comp_msg)
            claims_payload.append({
                "opponent_module_id": comp_id,
                "opponent_label": label,
                "claims": claims,
            })
            lines.append(f"- {label} claims:")
            lines.append(self._format_claims(claims))

        if shock_event:
            shock_type, shock_text = self._extract_shock_fields(shock_event)
            lines.append(f"- Shock payload [{shock_type}]: {shock_text[:1000]}")

        text = "\n".join(lines)
        return text[:max_chars], claims_payload

    def build_attack_vector(
        self,
        state: Dict[str, Any],
        current_module_id: str,
        max_chars: int = 2500,
    ) -> str:
        """Build Dual-Opponent Attack Vector A_opp for CFR Step A input."""
        text, _ = self.build_attack_vector_with_claims(
            state=state,
            current_module_id=current_module_id,
            max_chars=max_chars,
        )
        return text

    def build_module_perspective(
        self,
        state: Dict[str, Any],
        current_module_id: str,
        max_chars: int = 800,
    ) -> str:
        """Build a brief module perspective for the current speaker."""
        modules = state.get("modules", {})
        module_labels = state.get("module_labels") or self.build_module_labels(modules)
        label = module_labels.get(current_module_id, "MODULE")
        shock_event = state.get("shock_event") or state.get("perturbation")

        shock_line = ""
        if shock_event:
            shock_type, shock_text = self._extract_shock_fields(shock_event)
            shock_line = f"Shock context ({shock_type}): {shock_text[:400]}"

        perspective = f"{label} perspective. {shock_line}".strip()
        return perspective[:max_chars]
