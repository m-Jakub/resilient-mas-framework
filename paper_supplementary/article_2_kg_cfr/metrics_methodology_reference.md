## KES metrics: CC, ACA, PRR

### CC (Counterfactual Consistency)
- Location: scripts/analysis/compute_kes_metrics.py
- Inputs per turn:
  - private_strategy_nl (metadata key: private_strategy_nl)
  - public response (turn.message_content)
- Logic:
  - For each turn with both fields present, embed the private strategy and the public response using Gemini embeddings (models/gemini-embedding-001).
  - Compute cosine similarity, clamp to [0, 1].
  - Per-turn scores are averaged to avg_cc.

Key logic (abridged):
```python
private_strategy = metadata.get("private_strategy_nl") or metadata.get("private_strategy") or ""
public_response = turn.get("message_content") or ""
embeddings = model.embed_documents([private_strategy, public_response])
cc = cosine(emb_private, emb_public)
```

### ACA (Axiom-Claim Alignment)
- Location: scripts/analysis/compute_kes_metrics.py
- Axioms are loaded from identity graphs and assembled as:
  - up to 8 core beliefs
  - up to 5 values
  - up to 3 traits
- The ACA prompt asks a strict Yes/No question: whether the response is broadly consistent with the axioms.
- Parsing is minimal: checks first token and presence of yes/no.
- ACA is 1.0 for Yes, 0.0 for No; otherwise skipped.

Key logic (abridged):
```python
prompt = _build_aca_prompt(axioms, response)
raw = llm.invoke(prompt).content
consistent = _parse_yes_no(raw)
aca_scores[idx] = 1.0 if consistent else 0.0
```

### PRR (Perturbation Rebound Rate)
- Location: scripts/analysis/compute_kes_metrics.py
- PRR is computed only when a shock turn t has all of the following judge scores:
  - s_{t-1}, s_t, s_{t+1}, s_{t+2}
- And a quality drop threshold is satisfied:
  - drop = s_t - s_{t-1}
  - if drop >= -0.1, PRR is NOT computed for that shock.
- Formula:
  - prr = (s_{t+2} - s_t) / |drop|

Key logic (abridged):
```python
s_prev = by_turn.get(t-1)
s_t = by_turn.get(t)
s_t1 = by_turn.get(t+1)
s_t2 = by_turn.get(t+2)
if any is None: continue

drop = s_t - s_prev
if drop >= -0.1: continue
prr = (s_t2 - s_t) / max(abs(drop), EPS)
```

### Why n_prr drops for kg_cfr_full
From code semantics, n_prr (count of PRR rows) shrinks when either condition holds:
1) Missing judge scores for t-1, t, t+1, t+2 (any missing -> PRR not counted).
2) Shock does not cause a sufficiently negative drop (drop >= -0.1), so the PRR window is ignored.

In kg_cfr_full, the executor response is shaped by CFR strategy and KG retrieval. If the shock impact is muted (less negative drop), the PRR window is skipped, reducing n_prr. No regex is involved in PRR; it is purely judge-score window logic tied to shock turns and the drop threshold.

## Paper-specific reanalysis and spot-check artifacts

### PRR threshold reanalysis (-0.20)
- Location: scripts/analysis/recalc_prr_threshold.py
- Purpose: recompute PRR activation counts with a stricter drop threshold (drop <= -0.20).
- Output example: analysis/outputs/e2/prr_activation_threshold_-0.20.csv

### PRR by scenario/condition (pivot)
- Location: scripts/analysis/prr_by_scenario.py
- Inputs: corrected judge turns CSV with shock metadata.
- Output example: analysis/outputs/e2/prr_by_scenario_threshold_-0.20.csv

### Figure 4 (resilience summary) update
- Location: scripts/analysis/plot_figure4_resilience_updated.py
- Inputs:
  - logs/corrected_e2_batch_20260308_225734/summary/corrected_e2_aggregate.json
  - analysis/outputs/e2/prr_by_scenario_threshold_-0.20.csv
- Output: analysis/outputs/figures/figure4_resilience_summary.pdf

### Human spot-check sampling (blind evaluation)
- Location: scripts/analysis/extract_spotcheck_sample.py
- Outputs: human_spotcheck_blind.csv, human_spotcheck_key.csv

## DIS (D_proc)
- Location: scripts/analysis/compute_dis.py
- DIS is concept overlap between the current turn and the last opponent turn.
- concepts_mentioned is logged per turn; if missing, DIS cannot be computed.
- Default mode: Jaccard similarity:
  - DIS = |C_opponent ∩ C_current| / |C_opponent ∪ C_current|
- Alternate mode: opponent_recall:
  - DIS = |C_opponent ∩ C_current| / |C_opponent|

Key logic (abridged):
```python
opponent_turn = _find_last_opponent_turn(...)
current = _concept_set(turn)
opponent = _concept_set(opponent_turn)
if mode == "opponent_recall":
    dis = len(opponent & current) / max(1, len(opponent))
else:
    dis = len(opponent & current) / max(1, len(opponent | current))
```

## E2 runner N=30 pools

### Publication runner (run_corrected_e2.py)
- Target N=30 per condition
- CFR conditions use pooling: existing N=10 + TOPUP_N=20
- no_cfr_baseline uses BASELINE_N=30 fresh

Key constants:
```python
TOPUP_N = 20
BASELINE_N = 30
EXISTING = {"cfr_no_kg": "exp_e2_...", "kg_cfr_full": "exp_e2_..."}
```

### Frozen batch (run_corrected_e2_batch.py)
- TARGET_N = 30 per condition
- All sessions are pre-registered; TOPUP_N is 0 for all conditions
- Preflight checks enforce exact expected counts and KG activation

### Ad-hoc benchmark (run_e2_benchmark.py)
- Not the publication runner
- Default --runs is 50 (not N=30)
- Produces per-run aggregates and metrics, used for quick checks
