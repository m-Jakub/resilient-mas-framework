# Turn Pipeline & RAG Gating Architecture
**Status:** Current (verified against code on 2026-07-08)
**Reviewed:** 2026-07-08
**Verified against code:** 2026-07-08 ([turn_pipeline.py](src/agents/turn_pipeline.py#L407-L430), [philosopher_agents.py](src/agents/philosopher_agents.py#L491-L512), [aegis_orchestrator.py](src/agents/aegis_orchestrator.py#L131-L151))
**Last updated:** 2026-07-08

### TurnController behavior for kg_cfr_full
- Location: [src/agents/turn_pipeline.py](src/agents/turn_pipeline.py#L356-L430)
- Orchestrator constructs TurnController with canonical cfr_mode, and DebateState carries use_id_rag and use_adaptive_idrag flags.
- In TurnController.run_turn(), when cfr_mode == "kg_cfr_full":
  - A KGGateway is instantiated and passed into StrategyPlanner.plan(), enabling CFR to call RAG during strategy generation.
  - Double-RAG is prevented for the executor by forcing ID-RAG off at response time:
    - override_id_rag = False
    - override_adaptive = False
  - This means identity-graph RAG is used only in CFR planning (kg_cfr_full) and never during the executor response generation for that same turn.

Key logic (abridged):
```python
kg_gateway = KGGateway(module) if self.cfr_mode == "kg_cfr_full" else None
planner_output, kg_result = self.strategy_planner.plan(..., rag_gateway=kg_gateway)

override_id_rag = False if self.cfr_mode == "kg_cfr_full" else debate_state.use_id_rag
override_adaptive = False if self.cfr_mode == "kg_cfr_full" else debate_state.use_adaptive_idrag

response = self.response_composer.compose(...,
    use_id_rag=override_id_rag,
    use_adaptive_idrag=override_adaptive,
)
```

### KGGateway fallback: stop-words + longest tokens
- Location: [src/agents/turn_pipeline.py](src/agents/turn_pipeline.py#L210-L260)
- If query contains any ETHICAL_CONCEPTS or AEGIS_KEYWORDS substrings, those matched keywords are used directly.
- Otherwise, fallback is activated:
  - Lowercase query.
  - Split on whitespace, strip punctuation per token.
  - Remove stop-words set: {the, a, an, is, are, to, of, in, that, for, with, on, what, how, this, it}.
  - Rank remaining candidates by token length descending, then by original index (stable earliest).
  - Select the first 5 tokens as search terms.
  - If no evidence retrieved, fallback to identity_graph.get_core_beliefs().

Key logic (abridged):
```python
matched = [k for k in KEYWORDS if k in query_lc]
if matched:
    search_terms = matched
else:
    stop_words = {...}
    tokens = [t.strip(string.punctuation) for t in query_lc.split()]
    candidates = [(idx, t) for idx, t in enumerate(tokens) if t and t not in stop_words]
    candidates.sort(key=lambda item: (-len(item[1]), item[0]))
    search_terms = [term for _, term in candidates[:5]]
```

### Prompt engineering & recency bias
- Location: [src/agents/philosopher_agents.py](src/agents/philosopher_agents.py#L460-L512)
- The executor prompt is built with ChatPromptTemplate using ordered messages:
  1) system: full system scaffold + <persona_identity_axioms> block
  2) placeholder: chat_history
  3) human: question
  4) system: URGENT OVERRIDE FOR THIS TURN ONLY with <cfr_turn_strategy>

Key prompt structure (abridged):
```text
system:
  ...
  <persona_identity_axioms>
  {identity_context}
  </persona_identity_axioms>
  ...
  --- RETRIEVED CONTEXT (Axiomatic Corpus) ---
  {context}
  ...
  CRISIS CONTEXT ANCHOR
  {topic}

placeholder: {chat_history}
human: {question}
system:
  URGENT OVERRIDE FOR THIS TURN ONLY: ...
  <cfr_turn_strategy>
  {private_strategy}
  </cfr_turn_strategy>
```

- The ordering ensures CFR strategy is the last system message (recency bias fix) and does not appear earlier in the system scaffold.
- ID-RAG injection is controlled by use_id_rag/use_adaptive_idrag in respond(); when disabled (kg_cfr_full executor override), identity_context is empty and no ID-RAG retrieval is performed for the response.

### Metadata emission (audit-relevant)
- Location: [src/agents/turn_pipeline.py](src/agents/turn_pipeline.py#L300-L340)
- TurnLogger builds metadata including:
  - private_strategy_json, private_strategy_nl
  - cfr_generated, cfr_injected
  - rag_used, rag_docs_count
  - cf_flags
  - kg_status, kg_latency_ms, kg_fallback_reason
  - legacy compatibility keys: privatestrategyjson, cfrgenerated, ragdocscount, cfflags

Key metadata fields (abridged):
```python
metadata = {
  "private_strategy_json": ..., "private_strategy_nl": ...,
  "cfr_generated": ..., "cfr_injected": ...,
  "rag_used": ..., "rag_docs_count": ...,
  "cf_flags": ...,
  "kg_status": ..., "kg_latency_ms": ..., "kg_fallback_reason": ...,
  "privatestrategyjson": ..., "cfrgenerated": ..., "ragdocscount": ..., "cfflags": ...,
}
```

### Verification Log
- 2026-07-08: Verified `kg_cfr_full` wiring and executor RAG suppression in [src/agents/turn_pipeline.py](src/agents/turn_pipeline.py#L407-L430).
- 2026-07-08: Verified prompt ordering and final CFR override in [src/agents/philosopher_agents.py](src/agents/philosopher_agents.py#L491-L512) and the ID-RAG gate in [src/agents/philosopher_agents.py](src/agents/philosopher_agents.py#L787-L795).
- 2026-07-08: Verified canonical `cfr_mode` normalization and controller construction in [src/agents/aegis_orchestrator.py](src/agents/aegis_orchestrator.py#L131-L151).
- 2026-07-08: Verified KGGateway fallback behavior, including stop-word pruning, top-5 longest-token selection, and `get_core_beliefs()` fallback in [src/agents/turn_pipeline.py](src/agents/turn_pipeline.py#L210-L260).