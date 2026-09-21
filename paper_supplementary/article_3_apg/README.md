# kg_cfr_full corpus

This directory is a self-contained corpus of multi-agent debate trajectories.
It contains 90 knowledge-grounded trajectories across 3 crisis scenarios
(aegis_blackout, cerberus_biocontainment, synapse_orbital_strike),
with 30 runs per scenario. The runs were generated with a deterministic
seed for reproducibility and are used across ongoing research threads.

## Contents

- trajectories/
  - {scenario}/
    - kg_cfr_full/
      - run_XX.json

## Scenarios

- aegis_blackout
- cerberus_biocontainment
- synapse_orbital_strike

## Notes

- Each file is a single trajectory JSON with session metadata and turn logs.
- The condition folder is fixed as kg_cfr_full for all runs.
- Filenames are normalized to run_00.json through run_29.json per scenario.

## Phase 4: Active Provenance Gate (APG) Implementation
The complete runtime orchestration code, LangGraph state machine, self-healing fallback loop, and exact system/auditor prompt templates for Phase 4 are available in [`apg_subgraph.py`](apg_subgraph.py).

## Human-Centric Study Data (A/B Testing)
Raw, anonymized survey responses from our $N=33$ blind A/B study are provided in [`human_study_results_n33.csv`](human_study_results_n33.csv). This dataset evaluates calibrated trust and decision utility between the baseline synthesis and our APG Divergence Reports (Scenario $S_{009}$ and $S_{026}$).

*Note: The survey responses and questions were originally collected in Polish for local institutional testing, and have been preserved in their native form for full data provenance.*
