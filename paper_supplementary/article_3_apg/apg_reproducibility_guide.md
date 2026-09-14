# APG (Active Provenance Gate) - implementation and parameters

Verdict: verified against the current codebase on 2026-07-08.

This document collects implementation-level details about Active Provenance Gate (APG) and how the multi-agent synthesis phase is executed in the repository. It covers LLM hyperparameters, prompt text, thresholds, logging, and notes on text segmentation. Code references are provided in each section.

## 1) APG in the pipeline

APG runs after the main debate ends (tripartite standoff). The orchestrator extracts the LLM from the first module, aggregates axioms and private traces, then builds the synthesis graph and invokes it as a separate Phase 4.

Sources:
- [src/kg_cfr/aegis_orchestrator.py](src/kg_cfr/aegis_orchestrator.py) (method `_run_phase4_synthesis`, `build_synthesis_subgraph`, `recursion_limit`)
- [src/kg_cfr/aegis_orchestrator.py](src/kg_cfr/aegis_orchestrator.py) (log: `PHASE 4: ACTIVE PROVENANCE VERIFICATION LOOP`)

## 2) Core APG constants and thresholds

- `MAX_SYNTHESIS_ITER = 3`
- `PROVENANCE_FIDELITY_THRESHOLD = 0.95`

Sources:
- [src/apg/phase4_synthesis.py](src/apg/phase4_synthesis.py)

## 3) State structures and provenance logs

APG keeps synthesis state in `SynthesisState` and records a per-iteration provenance log (`ProvenanceRecord` list). The log includes, among other fields, the auditor verdict, `provenance_fidelity`, `unsupported_sentences`, iterations, and `token_overhead`.

Sources:
- [src/apg/phase4_synthesis.py](src/apg/phase4_synthesis.py) (`ProvenanceRecord`)
- [src/apg/phase4_synthesis.py](src/apg/phase4_synthesis.py) (`SynthesisState`)
- [src/apg/phase4_synthesis.py](src/apg/phase4_synthesis.py) (record creation in the validator)

## 4) Proposer and Validator (`SynthesisProposer` / `SynthesisValidator`)

- The proposer generates a policy using `crisis_context`, axioms, private traces, and debate history.
- The validator performs NLI/provenance evaluation (sentence-by-sentence assessment of the synthesis) and returns JSON with `verdict`, `provenance_fidelity`, `unsupported_sentences`, and `reasoning`.
- If `provenance_fidelity < threshold`, the result is forced to `rejected` even when the model returned `approved`.
- Iterations run for at most `MAX_SYNTHESIS_ITER`; afterward, a divergence report may be returned.

Sources:
- [src/apg/phase4_synthesis.py](src/apg/phase4_synthesis.py) (`SynthesisProposer`)
- [src/apg/phase4_synthesis.py](src/apg/phase4_synthesis.py) (`SynthesisValidator`, PF threshold)
- [src/apg/phase4_synthesis.py](src/apg/phase4_synthesis.py) (iteration router)
- [src/apg/phase4_synthesis.py](src/apg/phase4_synthesis.py) (divergence report)

## 5) APG prompts (exact templates)

### 5.1) Global Auditor - NLI/provenance prompt

System prompt:
```
_EVAL_CF_SYSTEM = (
    "You are a deterministic provenance auditor. "
    "Verify whether each sentence in the TEXT is grounded in the evidence provided. "
    "Respond with strict JSON only."
)
```

Template:
```
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
```

Sources:
- [src/apg/phase4_synthesis.py](src/apg/phase4_synthesis.py)

### 5.2) Executive Proposer - consensus prompt

System prompt:
```
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
```

Template:
```
_PROPOSER_TEMPLATE = (
    "CRISIS CONTEXT:\n{crisis_context}\n\n"
    "AXIOMATIC CONSTRAINTS:\n{axioms_block}\n\n"
    "PRIVATE TRACES:\n{private_traces_block}\n\n"
    "DEBATE HISTORY (last {history_turns} turns):\n{history_block}\n\n"
    "{correction_hint}"
    "Write a definitive, unified, operational policy synthesis that resolves the crisis. "
    "Respond in 3-5 sentences."
)
```

Sources:
- [src/apg/phase4_synthesis.py](src/apg/phase4_synthesis.py)

### 5.3) Correction hint (self-healing loop)

When `iteration > 0` and the validator returns `unsupported_sentences`, the proposer receives a correction hint:

```
CORRECTION REQUIRED (iteration {iteration}): The previous proposal was rejected because these sentences had no axiom support:
  {hint}
Remove or ground these claims. Stay within the axioms.
```

Sources:
- [src/apg/phase4_synthesis.py](src/apg/phase4_synthesis.py)

## 6) Text segmentation (sentences for NLI)

APG does not perform deterministic sentence segmentation in the APG code itself. The auditor prompt instructs the model to evaluate “each sentence in TEXT,” so sentence splitting is implicit and handled by the model.

There is a separate deterministic splitter used elsewhere in the repository (for example, metrics), based on a regex:
- `re.split(r"(?<=[.!?])\s+", normalized)`

Sources:
- [src/apg/phase4_synthesis.py](src/apg/phase4_synthesis.py) (auditor instruction about “each sentence”)
- [src/common/utils/claim_segmenter.py](src/common/utils/claim_segmenter.py) (regex split)

## 7) LLM hyperparameters and API configuration

Default LLM settings are defined in `config/settings.py`:
- `LLM_MODEL = "gemini-2.5-flash-lite"`
- `TEMPERATURE = 0.5`
- `MAX_TOKENS = 8192`

Google Gemini is instantiated as `ChatGoogleGenerativeAI` with:
- `model = model_name` (default or override)
- `temperature = TEMPERATURE`
- `max_output_tokens = MAX_TOKENS`

The code does not set `top_p` or other sampling parameters explicitly, so provider defaults apply.

Sources:
- [config/settings.py](config/settings.py#L2-L4)
- [src/common/api_abstraction.py](src/common/api_abstraction.py)

## 8) Separate proposer/auditor models + CLI overrides

The batch evaluator allows different models for proposer and auditor:
- `--proposer-model` (e.g. `gemini-3-flash-preview`)
- `--validator-model` (e.g. `gemini-3.1-pro-preview`)
- `--validator-threshold` (e.g. `0.95`)

Sources:
- [scripts/analysis/evaluate_provenance_batch.py](scripts/analysis/evaluate_provenance_batch.py#L426-L436)
- [scripts/analysis/evaluate_provenance_batch.py](scripts/analysis/evaluate_provenance_batch.py#L512-L525) (LLM creation and `build_synthesis_subgraph`)

## 9) Threshold logic and forced rejection

If the validator returns `approved` but `provenance_fidelity < threshold`, the result is overridden to `rejected` and a note about the override is appended.

Sources:
- [src/apg/phase4_synthesis.py](src/apg/phase4_synthesis.py)

## 10) Result logging and provenance JSONL

The batch evaluator writes:
- CSV metrics (`provenance_batch_*.csv`)
- JSONL per-iteration provenance records (`provenance_log_*.jsonl`)
- Audit trail (`audit_trails_*.txt`)

The JSONL contains, among other fields: `session_id`, `iteration`, `auditor_verdict`, `provenance_fidelity`, `proposed_text`, and `unsupported_sentences`.

Sources:
- [scripts/analysis/evaluate_provenance_batch.py](scripts/analysis/evaluate_provenance_batch.py#L540-L606) (provenance record construction)
- [scripts/analysis/evaluate_provenance_batch.py](scripts/analysis/evaluate_provenance_batch.py#L606-L637) (JSONL writing)
- [scripts/analysis/evaluate_provenance_batch.py](scripts/analysis/evaluate_provenance_batch.py#L639-L711) (audit trail writing)

## 11) Token estimation (`token_overhead`)

Token overhead is estimated with `tiktoken` (`cl100k_base`). If `tiktoken` is unavailable, the approximation `len(text)//4` is used.

Sources:
- [src/apg/phase4_synthesis.py](src/apg/phase4_synthesis.py)

## 12) Reproducibility summary

- APG is a separate Phase 4 run after the debate, with its own proposer/validator graph.
- The proposer creates a consensus synthesis from `crisis_context + axioms + private_traces + history`.
- The auditor evaluates each sentence against evidence (`history`/`traces`/`axioms`) and returns JSON with `verdict` and `unsupported_sentences`.
- The PF threshold is enforced strictly: `PF < 0.95` forces rejection.
- Maximum of 3 iterations, then a divergence report.
- LLM parameters: `temperature=0.5`, `max_output_tokens=8192`, configurable model; `top_p` is not explicitly set.
- Sentence segmentation is not explicit in APG code (the model handles it), but the repo includes a deterministic regex splitter for metrics.
