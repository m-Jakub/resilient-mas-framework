"""
Policy modules for the AEGIS/DRAU tripartite standoff.

Each module is an autonomous decision policy with a hidden axiomatic corpus
(backed by Identity-RAG) and a public role definition that does NOT reveal
its philosophical source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.common.philosopher_agents import create_agent, PhilosopherAgent


@dataclass
class ModuleProfile:
    module_id: str
    module_label: str
    objective: str
    identity_label: str
    hidden_source: str


@dataclass
class PolicyModule:
    module_id: str
    label: str
    objective: str
    agent: PhilosopherAgent

    def respond(
        self,
        crisis_context: str,
        attack_vector: str,
        shock_context: str,
        chat_history: list,
        ontology_insights: Optional[str] = None,
        context: str = "",
        private_strategy: str = "",
        use_id_rag: bool = True,
        use_adaptive_idrag: bool = True,
    ) -> str:
        """Generate a public response using the hidden axiomatic corpus."""
        question = (
            "You are drafting a policy response for the AEGIS committee.\n"
            f"Crisis context:\n{crisis_context}\n\n"
            f"Dual-opponent attack vector (A_opp):\n{attack_vector}\n\n"
            f"Telemetry shock:\n{shock_context}\n\n"
            "Task: Provide a concise, reasoned policy stance that defends your module's objective "
            "while directly addressing both opponents and the telemetry update."
        )

        return self.agent.respond(
            topic=crisis_context,
            chat_history=chat_history,
            ontology_insights=ontology_insights,
            use_id_rag=use_id_rag,
            use_adaptive_idrag=use_adaptive_idrag,
            private_strategy=private_strategy,
            question_override=question,
        )


def create_policy_module(profile: ModuleProfile, chroma_dir: str = "data/chroma") -> PolicyModule:
    """Factory for a policy module with a hidden philosophical corpus."""
    module_profile = {
        "public_name": profile.module_label,
        "public_role": profile.objective,
        "identity_label": profile.identity_label,
        "hidden_source": profile.hidden_source,
    }
    agent = create_agent(profile.hidden_source, chroma_dir=chroma_dir, module_profile=module_profile)
    if agent is None:
        raise ValueError(f"Failed to create module agent from source: {profile.hidden_source}")
    return PolicyModule(
        module_id=profile.module_id,
        label=profile.module_label,
        objective=profile.objective,
        agent=agent,
    )
