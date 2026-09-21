# Trajectory schema

Each trajectory file is a single JSON object with the following structure.
Fields may include additional keys not listed here; consumers should ignore
unknown fields to remain forward-compatible.

## Top-level object

- session_info: object
- turn_logs: array of objects

## session_info

- session_id: string
- crisis_context: string
- session_start: string (ISO 8601)
- metadata: object
- statistics: object

### session_info.metadata

- debate_turns: integer
- shock_type: string
- cfr_mode: string
- cfr_mode_legacy: string
- dilemma_id: string (scenario id)
- use_id_rag: boolean
- use_adaptive_idrag: boolean
- condition_label: string
- condition_label_legacy: string

### session_info.statistics

- total_turns: integer
- total_shocks: integer

## turn_logs[]

Each element is a single turn record.

- turn_number: integer
- timestamp: string (ISO 8601)
- phase: string
- module_id: string
- module_label: string
- message_content: string
- message_length: integer
- concepts_mentioned: array of strings
- response_time_ms: integer or null
- metadata: object

### turn_logs[].metadata (common keys)

- cf_flags: object
  - activated: boolean
  - turn_idx: integer
- private_strategy_json: string
- private_strategy_nl: string
- cfr_generated: boolean
- cfr_injected: boolean
- rag_used: boolean
- rag_docs_count: integer
- opponent_claims: array of objects
  - opponent_module_id: string
  - opponent_label: string
  - claims: array of objects
    - claim_id: string
    - claim_text: string
- kg_status: string
- kg_latency_ms: integer or null
- kg_fallback_reason: string
- planner_version: string
- profile_name: string
