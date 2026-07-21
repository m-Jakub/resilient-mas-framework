#!/usr/bin/env python3
"""
E2 Statistical Analysis — SCQA8_v8 Implementation
==================================================
Two-stage analysis per SCQA8_v8:

  STAGE 1  — LMMs for clarity, cogency, relevance
             Main test: Condition x TurnIndex interaction
             Random structure: (1 + TurnIndex | DebateID) + (1 | ScenarioID)

  STAGE 2  — Reconstruct Dproc = z(SR) - z(DIS), then same LMM

  COMPLEMENTARY (E2 resilience metrics, separate from Stage 1/2):
    * DIS and ACA — shock-stratified descriptives + LMM
    * PRR         — shock-stratified descriptives (N too small for LMM)
    * DeltaCC     — report aligned cc_v5 value per condition;
                    NOTE: the Adversarial Swapped-Plan Test (DeltaCC as defined
                    in SCQA8_v8 S4) requires raw embedding vectors (v_Pt, v_Et)
                    that are not present in E2_Turns_judge_fixed.csv.
                    cc_v5 is the ALIGNED score only; swapped baseline cannot be
                    computed from this CSV. We therefore report aligned CC
                    descriptives and the LMM on cc_v5, but explicitly do NOT
                    claim this constitutes the Swapped-Plan Test.

  S4 TOST  — Two One-Sided Tests for equivalence on the negative-control shock.
             Equivalence margin = +-0.5 pooled SD.

  FDR      — Benjamini-Hochberg across all non-intercept fixed effects in the
             four core models (clarity, cogency, relevance, dproc).

Input : logs/corrected_e2_batch_20260308_225734/E2_Turns_judge_fixed.csv
Output: analysis/outputs/e2/e2_scqa8_results.txt   (full report)
        analysis/outputs/e2/e2_scqa8_coefs.csv      (per-coefficient table)
        analysis/outputs/e2/e2_scqa8_fdr.csv        (FDR-corrected p-values)
        analysis/outputs/e2/e2_scqa8_tost.csv       (S4 TOST results)
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy import stats

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Covariance.*")
warnings.filterwarnings("ignore", message=".*convergence.*")

# --- paths -------------------------------------------------------------------
ROOT    = Path(__file__).resolve().parents[2]
CSV_IN_DEFAULT = ROOT / "logs" / "corrected_e2_batch_20260308_225734" / "E2_Turns_judge_fixed.csv"
CSV_IN = Path(os.getenv("SCQA8_CSV_IN", str(CSV_IN_DEFAULT)))
OUT_DIR = ROOT / "analysis" / "outputs" / "e2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_F = OUT_DIR / "e2_scqa8_results.txt"
COEF_F   = OUT_DIR / "e2_scqa8_coefs.csv"
FDR_F    = OUT_DIR / "e2_scqa8_fdr.csv"
TOST_F   = OUT_DIR / "e2_scqa8_tost.csv"

# --- logging -----------------------------------------------------------------
_lines: list = []

def log(msg: str = ""):
    print(msg)
    _lines.append(msg)

def section(title: str):
    bar = "=" * 72
    log(f"\n{bar}")
    log(f"  {title}")
    log(bar)

def sub(title: str):
    log(f"\n  -- {title} --")

def save_report():
    REPORT_F.write_text("\n".join(_lines), encoding="utf-8")


# --- LMM helpers -------------------------------------------------------------
# SCQA8_v8 spec:
#   Score ~ Condition + ShockType + TurnIndex + Condition:TurnIndex
#           + (1 + TurnIndex | DebateID) + (1 | ScenarioID)
#
# With only 1 unique ScenarioID the scenario RE is degenerate.
# We attempt it and fall back gracefully, recording the reason.

def _lmm_full(formula: str, data: pd.DataFrame, label: str):
    """
    Attempt full SCQA8_v8 random structure, fall back progressively.
    1. (1 + turn_c | debate_id) + (1 | scenario_id)
    2. (1 + turn_c | debate_id)
    3. (1 | debate_id)
    Returns (fitted result, re_type string).
    """
    n_scen = data["scenario_id"].nunique()
    n_deb  = data["debate_id"].nunique()
    log(f"    debates={n_deb}, scenarios={n_scen}")

    # Attempt 1: full spec (random slope + scenario RE)
    if n_scen > 1:
        try:
            vc  = {"scenario_id": "0 + C(scenario_id)"}
            m   = smf.mixedlm(formula, data=data, groups=data["debate_id"],
                              re_formula="~turn_c", vc_formula=vc)
            res = m.fit(reml=True, maxiter=300)
            if res.converged:
                log(f"    [{label}] FULL spec converged (random slope + scenario RE)")
                return res, "slope+scenario"
        except Exception as e:
            log(f"    [{label}] full spec failed ({e})")
    else:
        log(f"    [{label}] Only {n_scen} scenario(s) -> (1|ScenarioID) degenerate; "
            f"dropping scenario RE (single-scenario design, SCQA8_v8 §E0)")

    # Attempt 2: random slope only
    try:
        m   = smf.mixedlm(formula, data=data, groups=data["debate_id"],
                          re_formula="~turn_c")
        res = m.fit(reml=True, maxiter=300)
        if res.converged:
            log(f"    [{label}] random-slope converged (no scenario RE)")
            return res, "slope_only"
        log(f"    [{label}] random-slope did not converge; trying intercept-only")
    except Exception as e:
        log(f"    [{label}] random-slope failed ({e}); trying intercept-only")

    # Attempt 3: random intercept only
    m   = smf.mixedlm(formula, data=data, groups=data["debate_id"])
    res = m.fit(reml=True, maxiter=300)
    log(f"    [{label}] intercept-only converged={res.converged}")
    return res, "intercept_only"


def fmt_model_table(res) -> str:
    return res.summary().tables[1].to_string()


def result_row(dv: str, coef: str, res, re_type: str) -> dict:
    return {
        "dv":       dv,
        "coef":     coef,
        "estimate": res.fe_params[coef],
        "se":       res.bse[coef],
        "z":        res.tvalues[coef],
        "p_raw":    res.pvalues[coef],
        "p_fdr":    np.nan,
        "re_type":  re_type,
    }


# --- TOST helper -------------------------------------------------------------
def tost(g1: np.ndarray, g2: np.ndarray,
         margin_sd: float = 0.5, label: str = "") -> dict:
    """Welch-based Two One-Sided Tests for equivalence."""
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    if len(g1) < 3 or len(g2) < 3:
        return {"label": label, "n1": len(g1), "n2": len(g2),
                "p_tost": np.nan, "equiv": False, "note": "insufficient N"}
    v1, v2  = np.var(g1, ddof=1), np.var(g2, ddof=1)
    n1, n2  = len(g1), len(g2)
    pooled_sd = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2))
    delta     = margin_sd * pooled_sd
    diff      = g1.mean() - g2.mean()
    se_w      = np.sqrt(v1/n1 + v2/n2)
    df_w      = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))
    t_lo      = (diff - (-delta)) / se_w
    t_hi      = (diff -   delta)  / se_w
    p_lo      = stats.t.sf(t_lo, df_w)
    p_hi      = stats.t.cdf(t_hi, df_w)
    p_tost    = max(p_lo, p_hi)
    return {
        "label": label, "n1": n1, "n2": n2,
        "mean1": g1.mean(), "mean2": g2.mean(),
        "diff": diff, "pooled_sd": pooled_sd, "delta": delta,
        "t_lower": t_lo, "t_upper": t_hi,
        "p_lower": p_lo, "p_upper": p_hi,
        "p_tost": p_tost, "equiv": bool(p_tost < 0.05), "note": "",
    }


# =============================================================================
#  LOAD & PREPARE
# =============================================================================
section("DATA LOADING & PREPARATION")

df = pd.read_csv(CSV_IN)
log(f"  Loaded: {len(df)} rows | {df['debate_id'].nunique()} debates "
    f"| {df['condition'].nunique()} conditions "
    f"| {df['scenario_id'].nunique()} scenario(s)")
log(f"  Conditions : {sorted(df['condition'].unique())}")
log(f"  Scenario(s): {sorted(df['scenario_id'].unique())}")
log(f"  Turn range : {df['turn_number'].min()}-{df['turn_number'].max()}")
log(f"  Shocks     : {df['shock_type_turn'].value_counts(dropna=False).to_dict()}")

REF = "no_cfr_baseline"

df["shock_turn"] = df["shock_type_turn"].fillna("none")
turn_mean = df["turn_number"].mean()
df["turn_c"] = df["turn_number"] - turn_mean
log(f"  turn_c = turn_number - {turn_mean:.3f} "
    f"(range {df['turn_c'].min():.1f} to {df['turn_c'].max():.1f})")

df_j = df.dropna(subset=["clarity", "cogency", "relevance"]).copy()
log(f"  Judge-complete rows: {len(df_j)}")

df_shock = df_j[df_j["shock_turn"] != "none"].copy()
log(f"  Shock-present rows : {len(df_shock)} "
    f"({df_shock['shock_turn'].value_counts().to_dict()})")


# =============================================================================
#  DESCRIPTIVE STATISTICS
# =============================================================================
section("DESCRIPTIVE STATISTICS BY CONDITION")

for dv in ["clarity", "cogency", "relevance", "overall"]:
    sub(dv)
    tbl = df_j.groupby("condition")[dv].agg(
        N="count", mean="mean", sd="std",
        p25=lambda x: x.quantile(0.25),
        med="median",
        p75=lambda x: x.quantile(0.75),
    ).round(4)
    log(tbl.to_string())

sub("Mean judge scores at shock turns by condition")
log(df_shock.groupby(["condition", "shock_turn"])[
    ["clarity", "cogency", "relevance"]].mean().round(3).to_string())


# =============================================================================
#  STAGE 1 — LMMs: clarity, cogency, relevance
# =============================================================================
section("STAGE 1 -- LMMs: clarity / cogency / relevance")
log("""
  Model (SCQA8_v8 §E0):
    Score ~ C(condition, Treatment('no_cfr_baseline'))
          + C(shock_turn, Treatment('none'))
          + turn_c
          + C(condition, Treatment('no_cfr_baseline')):turn_c
          + (1 + turn_c | debate_id)
          + (1 | scenario_id)   [attempted; degenerate if n_scenarios=1]

  Main test: Condition:turn_c (differential slope over time).
  Reference condition = no_cfr_baseline.
  Reference shock     = none.
""")

FORMULA_CORE = (
    "{dv} ~ C(condition, Treatment(reference='{ref}'))"
    " + C(shock_turn, Treatment(reference='none'))"
    " + turn_c"
    " + C(condition, Treatment(reference='{ref}')):turn_c"
)

s1_results: dict = {}
all_coef_rows: list = []
all_pvals: dict = {}

for dv in ["clarity", "cogency", "relevance"]:
    sub(f"DV = {dv.upper()}")
    formula = FORMULA_CORE.format(dv=dv, ref=REF)
    res, re_type = _lmm_full(formula, df_j, dv)
    s1_results[dv] = (res, re_type)

    log(f"\n{fmt_model_table(res)}")
    log(f"  converged={res.converged}  |  random-effects type: {re_type}")
    log(f"  log-likelihood={res.llf:.3f}")

    for coef in res.fe_params.index:
        all_coef_rows.append(result_row(dv, coef, res, re_type))
        if "Intercept" not in coef:
            all_pvals[(dv, coef)] = res.pvalues[coef]


# =============================================================================
#  STAGE 2 — Dproc reconstruction + LMM
# =============================================================================
section("STAGE 2 -- Dproc Reconstruction & LMM")
log("""
  Dproc(t) = alpha * SR(t) - beta * DIS(t)

  Per SCQA8_v8 §Weights: 'parameters alpha and beta are calibrated on a
  development split in order to z-normalize contributions of SR and DIS.'

  *** IMPLEMENTATION NOTE ***
  compute_v5_metrics.py (the versioned metric source) computes SR and DIS
  as separate scores and does NOT include a Dproc composition function with
  fixed calibrated alpha/beta values.  No confirmed alpha/beta constants were
  found in the codebase.  This reconstruction therefore applies z-normalization
  to equate the units of SR and DIS (making both mean-0, sd-1 within this
  dataset) and sums them with equal weights as a proxy pending confirmed
  calibration.  Treat the resulting index as an equal-weight z-normalized
  proxy, NOT a verified implementation of the published calibrated weights.

  Columns: sr_v5 (available turns 4-11), dis_v5 (turns 2-11)
  Dproc requires BOTH -> eligible rows are turns 4-11 (720 / 90 debates).
""")

df_dp = df.dropna(subset=["sr_v5", "dis_v5"]).copy()
df_dp["shock_turn"] = df_dp["shock_type_turn"].fillna("none")
df_dp["turn_c"]     = df_dp["turn_number"] - turn_mean

df_dp["sr_z"]  = (df_dp["sr_v5"]  - df_dp["sr_v5"].mean())  / df_dp["sr_v5"].std()
df_dp["dis_z"] = (df_dp["dis_v5"] - df_dp["dis_v5"].mean()) / df_dp["dis_v5"].std()
df_dp["dproc"] = df_dp["sr_z"] - df_dp["dis_z"]   # proxy: equal-weight z-norm

log(f"  Dproc-eligible rows: {len(df_dp)} "
    f"(turns {df_dp['turn_number'].min()}-{df_dp['turn_number'].max()})")
log(f"  sr_v5  : mean={df_dp['sr_v5'].mean():.4f}  sd={df_dp['sr_v5'].std():.4f}")
log(f"  dis_v5 : mean={df_dp['dis_v5'].mean():.4f}  sd={df_dp['dis_v5'].std():.4f}")
log(f"  Dproc  : mean={df_dp['dproc'].mean():.4f}  sd={df_dp['dproc'].std():.4f}"
    f"  range=[{df_dp['dproc'].min():.3f}, {df_dp['dproc'].max():.3f}]")

sub("Dproc by condition")
log(df_dp.groupby("condition")["dproc"].describe().round(4).to_string())

sub("Shock types in Dproc subset (turns 4+)")
log(df_dp["shock_turn"].value_counts(dropna=False).to_string())

sub("DV = DPROC")
formula_dp = FORMULA_CORE.format(dv="dproc", ref=REF)

res_dp, re_dp = _lmm_full(formula_dp, df_dp, "dproc")
s1_results["dproc"] = (res_dp, re_dp)

log(f"\n{fmt_model_table(res_dp)}")
log(f"  converged={res_dp.converged}  |  random-effects type: {re_dp}")
log(f"  log-likelihood={res_dp.llf:.3f}")

for coef in res_dp.fe_params.index:
    all_coef_rows.append(result_row("dproc", coef, res_dp, re_dp))
    if "Intercept" not in coef:
        all_pvals[("dproc", coef)] = res_dp.pvalues[coef]


# =============================================================================
#  MAIN HYPOTHESIS SUMMARY: Condition x TurnIndex
# =============================================================================
section("MAIN TEST SUMMARY: Condition x TurnIndex Interaction")
log("  SCQA8_v8 §E0: 'report Condition:TurnIndex as the main test of "
    "differential degradation over time'")
log("")

for dv, (res, re_type) in s1_results.items():
    for coef in res.fe_params.index:
        if ":turn_c" in coef.lower() and "intercept" not in coef.lower():
            p   = res.pvalues[coef]
            sig = ("***" if p < 0.001 else "**" if p < 0.01
                   else "*" if p < 0.05 else "ns")
            log(f"  {dv:<10} {coef}")
            log(f"             beta={res.fe_params[coef]:+.5f}  "
                f"SE={res.bse[coef]:.5f}  "
                f"z={res.tvalues[coef]:+.3f}  "
                f"p_raw={p:.4f} ({sig})")


# =============================================================================
#  FDR CORRECTION (BH)
# =============================================================================
section("FDR CORRECTION (Benjamini-Hochberg)")
log("  Applied across all non-intercept fixed effects in "
    "clarity, cogency, relevance, dproc (4 models).")

fdr_keys   = list(all_pvals.keys())
fdr_raw    = np.array([all_pvals[k] for k in fdr_keys])

reject, p_adj, _, _ = multipletests(fdr_raw, alpha=0.05, method="fdr_bh")
fdr_map = {k: p_adj[i] for i, k in enumerate(fdr_keys)}

log(f"  Total tests: {len(fdr_raw)}   Rejected at FDR < 0.05: {reject.sum()}")

fdr_df = pd.DataFrame({
    "dv":     [k[0] for k in fdr_keys],
    "coef":   [k[1] for k in fdr_keys],
    "p_raw":  fdr_raw,
    "p_fdr":  p_adj,
    "sig_fdr": reject,
}).sort_values("p_fdr")

sig10 = fdr_df[fdr_df["p_fdr"] < 0.10]
log(f"\n  FDR < 0.10 ({len(sig10)} terms):")
if len(sig10) == 0:
    log("    (none)")
else:
    log(sig10.to_string(index=False))

log(f"\n  Full FDR table (sorted by p_fdr):")
log(fdr_df.to_string(index=False))

# Back-fill FDR p-values into coefficient rows
for row in all_coef_rows:
    row["p_fdr"] = fdr_map.get((row["dv"], row["coef"]), np.nan)

# Update main hypothesis summary with FDR p
sub("Condition x TurnIndex — raw vs FDR-corrected p")
for dv, (res, _) in s1_results.items():
    for coef in res.fe_params.index:
        if ":turn_c" in coef.lower() and "intercept" not in coef.lower():
            p_raw = res.pvalues[coef]
            p_fdr = fdr_map.get((dv, coef), np.nan)
            log(f"  {dv:<10} p_raw={p_raw:.4f}  p_fdr={p_fdr:.4f}  coef={coef}")


# =============================================================================
#  COMPLEMENTARY E2 METRICS
# =============================================================================
section("COMPLEMENTARY E2 METRICS  (SCQA8_v8 §E2 -- separate from Stage 1/2)")
log("""
  Per SCQA8_v8: DIS, ACA, PRR, and DeltaCC are analyzed as complementary E2
  metrics targeting responsiveness, doctrinal adherence, post-shock recovery,
  and planning-execution consistency under adversarial shocks S1-S4.
  They are NOT part of the Stage 1/2 longitudinal judge-quality core.

  Analysis scope:
    * Shock turns      : rows where shock_type_turn in {S1, S2, S3, S4}
    * Post-shock window: shock turn + up to 2 subsequent turns within the same
      debate (matching compute_kes_metrics.py _compute_prr window definition:
      scores at t, t+1, t+2 where t is the shock turn).
  This restriction is intentional: SCQA8_v8 §E2 binds these metrics to
  resilience under perturbation, not to general longitudinal trends.
""")


def _shock_window(base_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Return rows that fall within the post-shock window of any shock turn:
    shock turn t, t+1, t+2 within the same debate_id.
    Adds column 'window_offset' (0 = shock turn, 1 = t+1, 2 = t+2) and
    'anchor_shock_turn' (shock type of the originating shock).
    """
    base_df = base_df.dropna(subset=[value_col]).copy()
    base_df["shock_turn"] = base_df["shock_type_turn"].fillna("none")
    # Collect shock anchors: (debate_id, turn_number, shock_type)
    anchors = base_df[base_df["shock_turn"] != "none"][
        ["debate_id", "turn_number", "shock_turn"]
    ].rename(columns={"turn_number": "shock_t", "shock_turn": "anchor_shock"})
    rows = []
    for _, a in anchors.iterrows():
        window_turns = [a["shock_t"], a["shock_t"] + 1, a["shock_t"] + 2]
        for offset, wt in enumerate(window_turns):
            match = base_df[
                (base_df["debate_id"] == a["debate_id"]) &
                (base_df["turn_number"] == wt)
            ].copy()
            if len(match) == 0:
                continue
            match["window_offset"]   = offset
            match["anchor_shock"]    = a["anchor_shock"]
            rows.append(match)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# --- DIS ---------------------------------------------------------------------
sub("DIS (dis_v5) -- Dialectical Interactivity  [shock + post-shock window only]")
log("""
  Lightweight proxy for opponent-directed conceptual engagement.
  Reported for shock turns and the 2-turn post-shock window (t, t+1, t+2).
  SCQA8_v8 §E2 treats DIS as a resilience metric under shocks; longitudinal
  trends over all non-shock turns are not reported here.
""")

df_dis_w = _shock_window(df, "dis_v5")
log(f"  Shock-window DIS rows: {len(df_dis_w)}")

if len(df_dis_w) > 0:
    sub("DIS at shock turn (window_offset=0) by condition x shock type")
    dis_t0 = df_dis_w[df_dis_w["window_offset"] == 0]
    log(dis_t0.groupby(["condition", "anchor_shock"])["dis_v5"]
        .mean().round(4).unstack(fill_value=np.nan).to_string())

    sub("DIS mean across full post-shock window (offset 0-2) by condition x shock")
    log(df_dis_w.groupby(["condition", "anchor_shock"])["dis_v5"]
        .mean().round(4).unstack(fill_value=np.nan).to_string())

    sub("DIS mean by window_offset (0=shock, 1=t+1, 2=t+2) x condition")
    log(df_dis_w.groupby(["window_offset", "condition"])["dis_v5"]
        .mean().round(4).unstack(fill_value=np.nan).to_string())

    sub("DIS shock-window LMM: dis_v5 ~ C(condition) + C(anchor_shock) + window_offset + C(condition):window_offset")
    log("  (groups = debate_id; models DIS trajectory within the post-shock window)")
    df_dis_w["turn_c"] = df_dis_w["turn_number"] - turn_mean
    dis_sw_formula = (
        "dis_v5 ~ C(condition, Treatment(reference='{ref}'))"
        " + C(anchor_shock, Treatment(reference='S1'))"
        " + window_offset"
        " + C(condition, Treatment(reference='{ref}')):window_offset"
    ).format(ref=REF)
    try:
        res_dis, re_dis = _lmm_full(dis_sw_formula, df_dis_w, "dis_sw")
        log(f"\n{fmt_model_table(res_dis)}")
        log(f"  converged={res_dis.converged}  |  {re_dis}")
    except Exception as e:
        log(f"  DIS shock-window LMM failed: {e}")
else:
    log("  No shock-window DIS rows found.")

# --- ACA ---------------------------------------------------------------------
sub("ACA -- Axiomatic Constraint Adherence  [shock + post-shock window only]")
log("""
  Operational proxy for doctrinal constraint compliance.
  Reported for shock turns and the 2-turn post-shock window (t, t+1, t+2).
  SCQA8_v8 §E2 treats ACA as a resilience metric; non-shock turns excluded.
""")

df_aca_w = _shock_window(df, "aca")
log(f"  Shock-window ACA rows: {len(df_aca_w)}")

if len(df_aca_w) > 0:
    sub("ACA at shock turn (window_offset=0) by condition x shock type")
    aca_t0 = df_aca_w[df_aca_w["window_offset"] == 0]
    log(aca_t0.groupby(["condition", "anchor_shock"])["aca"]
        .mean().round(4).unstack(fill_value=np.nan).to_string())

    sub("ACA mean across full post-shock window by condition x shock")
    log(df_aca_w.groupby(["condition", "anchor_shock"])["aca"]
        .mean().round(4).unstack(fill_value=np.nan).to_string())

    sub("ACA mean by window_offset x condition")
    log(df_aca_w.groupby(["window_offset", "condition"])["aca"]
        .mean().round(4).unstack(fill_value=np.nan).to_string())

    sub("ACA shock-window LMM")
    df_aca_w["turn_c"] = df_aca_w["turn_number"] - turn_mean
    aca_sw_formula = (
        "aca ~ C(condition, Treatment(reference='{ref}'))"
        " + C(anchor_shock, Treatment(reference='S1'))"
        " + window_offset"
        " + C(condition, Treatment(reference='{ref}')):window_offset"
    ).format(ref=REF)
    try:
        res_aca, re_aca = _lmm_full(aca_sw_formula, df_aca_w, "aca_sw")
        log(f"\n{fmt_model_table(res_aca)}")
        log(f"  converged={res_aca.converged}  |  {re_aca}")
    except Exception as e:
        log(f"  ACA shock-window LMM failed: {e}")
else:
    log("  No shock-window ACA rows found.")

# --- PRR ---------------------------------------------------------------------
sub("PRR -- Perturbation Rebound Rate  [inherently shock-indexed]")
log("""
  Post-shock judge-score recovery metric.
  Defined in compute_kes_metrics.py _compute_prr as:
    drop = overall(t) - overall(t-1)    [requires drop < -0.1]
    PRR  = (overall(t+2) - overall(t)) / |drop|
    rebound_1 = overall(t+1) - overall(t)
    rebound_2 = overall(t+2) - overall(t+1)
  PRR is only computed for turns with a meaningful quality drop (>0.1);
  it is therefore already a shock-window measurement by construction.
""")

df_prr = df.dropna(subset=["prr"]).copy()
df_prr["shock_turn"] = df_prr["shock_type_turn"].fillna("none")

log(f"  PRR rows: {len(df_prr)}")
if len(df_prr) < 15:
    log(f"  *** NOTE: N={len(df_prr)} is insufficient for LMM *** "
        f"Descriptives and individual rows reported only.")

if len(df_prr) > 0:
    sub("PRR by condition")
    log(df_prr.groupby("condition")["prr"].describe().round(4).to_string())

    sub("PRR by condition x shock_turn")
    if df_prr["shock_turn"].nunique() > 1:
        log(df_prr.groupby(["condition", "shock_turn"])["prr"]
            .mean().round(4).unstack(fill_value=np.nan).to_string())
    else:
        log(df_prr.groupby("condition")["prr"].mean().round(4).to_string())

    sub("All PRR rows (shock turn, drop, and recovery trajectory)")
    log(df_prr[["condition", "debate_id", "turn_number", "shock_turn",
                 "prr", "prr_drop", "prr_rebound_1", "prr_rebound_2"]]
        .sort_values(["condition", "turn_number"])
        .to_string(index=False))

# --- CC (aligned only) -------------------------------------------------------
sub("CC -- Counterfactual Consistency (ALIGNED score)  [shock + post-shock window]")
log("""
  *** SCOPE LIMITATION — DO NOT CONFLATE WITH THE SWAPPED-PLAN TEST ***

  SCQA8_v8 §H3.2 defines:
    DeltaCC_t = sim(v_Pt, v_Et) - E_{i!=t}[sim(v_Pt, v_Ei)]
  where v_Pt = private plan embedding, v_Et = public response embedding.

  The Swapped-Plan Test REQUIRES raw embedding vectors (v_Pt, v_Ei for i!=t).
  These are NOT stored in E2_Turns_judge_fixed.csv.

  cc_v5 = aligned similarity only (numerator of DeltaCC; no swapped baseline).
  Analysis here:
    (a) Aligned CC over the shock + post-shock window — characterises
        plan-execution consistency specifically during adversarial perturbation.
    (b) cc_kes vs cc_v5 cross-check (embedding version stability, all CC rows).
  Neither (a) nor (b) constitutes the Swapped-Plan Test.
""")

df_cc_w = _shock_window(df, "cc_v5")
log(f"  Shock-window CC rows: {len(df_cc_w)}")

if len(df_cc_w) > 0:
    cc_conds_w = set(df_cc_w["condition"].unique())
    sub("Aligned CC at shock turn (offset=0) by condition x shock type")
    cc_t0 = df_cc_w[df_cc_w["window_offset"] == 0]
    if len(cc_t0) > 0:
        log(cc_t0.groupby(["condition", "anchor_shock"])["cc_v5"]
            .mean().round(4).unstack(fill_value=np.nan).to_string())
    else:
        log("  (no CC rows at shock turn itself)")

    sub("Aligned CC mean across post-shock window by condition x shock")
    log(df_cc_w.groupby(["condition", "anchor_shock"])["cc_v5"]
        .mean().round(4).unstack(fill_value=np.nan).to_string())

    if REF in cc_conds_w and len(cc_conds_w) >= 2:
        sub("Aligned CC shock-window LMM")
        df_cc_w["turn_c"] = df_cc_w["turn_number"] - turn_mean
        cc_sw_formula = (
            "cc_v5 ~ C(condition, Treatment(reference='{ref}'))"
            " + C(anchor_shock, Treatment(reference='S1'))"
            " + window_offset"
            " + C(condition, Treatment(reference='{ref}')):window_offset"
        ).format(ref=REF)
        try:
            res_cc, re_cc = _lmm_full(cc_sw_formula, df_cc_w, "cc_sw")
            log(f"\n{fmt_model_table(res_cc)}")
            log(f"  converged={res_cc.converged}  |  {re_cc}")
        except Exception as e:
            log(f"  CC shock-window LMM failed: {e}")
    else:
        log(f"  Reference condition '{REF}' absent from CC shock-window subset "
            f"({sorted(cc_conds_w)}); LMM not applicable.")
else:
    log("  No shock-window CC rows found.")

log("\n  (b) cc_kes vs cc_v5 embedding robustness cross-check (all CC turns):")
df_cc2 = df.dropna(subset=["cc_kes", "cc_v5"]).copy()
if len(df_cc2) > 0:
    r,   p_r   = stats.pearsonr(df_cc2["cc_kes"], df_cc2["cc_v5"])
    rho, p_rho = stats.spearmanr(df_cc2["cc_kes"], df_cc2["cc_v5"])
    log(f"  N={len(df_cc2)}")
    log(f"  Pearson  r={r:.4f}  (p={p_r:.4f})")
    log(f"  Spearman r={rho:.4f}  (p={p_rho:.4f})")
    log(f"  cc_kes: mean={df_cc2['cc_kes'].mean():.4f}  sd={df_cc2['cc_kes'].std():.4f}")
    log(f"  cc_v5 : mean={df_cc2['cc_v5'].mean():.4f}  sd={df_cc2['cc_v5'].std():.4f}")


# =============================================================================
#  S4 TOST
# =============================================================================
section("S4 TOST -- Resilience Equivalence on Negative-Control Shock (SCQA8_v8 §H3.1)")
log("""
  S4 = direct ad-hominem shock (emotional, non-logical negative control).
  SCQA8_v8 hypothesis: the KG-CFR architecture's protective effects are
  specific to logical resilience (S1-S3); on the non-logical S4 shock all
  conditions should be statistically EQUIVALENT in their resilience response.

  Primary metrics (resilience under S4):
    DIS at S4 shock turn — opponent-directed responsiveness under ad-hominem
    ACA at S4 shock turn — doctrinal constraint adherence under ad-hominem
    These directly measure process-level resilience behavior.

  Secondary metrics (judge dimensions at S4 turn, for completeness):
    overall, clarity, cogency, relevance
    These are included as supporting context, not as the primary resilience test.

  Equivalence margin: +-0.5 pooled SD.  Test: Two One-Sided Tests (Welch).
  Data: S4 shock turns only (shock_type_turn == 'S4').
""")

df_s4_j   = df_j[df_j["shock_turn"] == "S4"].copy()      # judge-complete S4 rows
df_s4_all = df[df["shock_type_turn"] == "S4"].copy()     # all S4 rows (for DIS/ACA)

log(f"  S4 judge rows : {len(df_s4_j)} | {df_s4_j['condition'].value_counts().to_dict()}")
log(f"  S4 all rows   : {len(df_s4_all)} | {df_s4_all['condition'].value_counts().to_dict()}")

tost_rows = []

# ── PRIMARY: resilience process metrics ──────────────────────────────────────
sub("PRIMARY — DIS at S4 shock turn")
for cond in ["cfr_no_kg", "kg_cfr_full"]:
    g1 = df_s4_all.loc[df_s4_all["condition"] == cond, "dis_v5"].values
    g2 = df_s4_all.loc[df_s4_all["condition"] == REF,  "dis_v5"].values
    r  = tost(g1, g2, margin_sd=0.5, label=f"DIS_S4: {cond} vs {REF}")
    tost_rows.append(r)
    verdict = "EQUIVALENT" if r["equiv"] else "not equivalent"
    log(f"\n  {r['label']}")
    if r.get("note") == "insufficient N":
        log(f"    n1={r['n1']}  n2={r['n2']}  *** INSUFFICIENT N ***")
    else:
        log(f"    n1={r['n1']}  n2={r['n2']}  "
            f"diff={r['diff']:+.4f}  pooled_sd={r['pooled_sd']:.4f}  "
            f"delta=+-{r['delta']:.4f}")
        log(f"    p_lower={r['p_lower']:.4f}  p_upper={r['p_upper']:.4f}  "
            f"p_TOST={r['p_tost']:.4f}  ->  {verdict} at alpha=0.05")

sub("PRIMARY — ACA at S4 shock turn")
for cond in ["cfr_no_kg", "kg_cfr_full"]:
    g1 = df_s4_all.loc[df_s4_all["condition"] == cond, "aca"].values
    g2 = df_s4_all.loc[df_s4_all["condition"] == REF,  "aca"].values
    r  = tost(g1, g2, margin_sd=0.5, label=f"ACA_S4: {cond} vs {REF}")
    tost_rows.append(r)
    verdict = "EQUIVALENT" if r["equiv"] else "not equivalent"
    log(f"\n  {r['label']}")
    if r.get("note") == "insufficient N":
        log(f"    n1={r['n1']}  n2={r['n2']}  *** INSUFFICIENT N ***")
    else:
        log(f"    n1={r['n1']}  n2={r['n2']}  "
            f"diff={r['diff']:+.4f}  pooled_sd={r['pooled_sd']:.4f}  "
            f"delta=+-{r['delta']:.4f}")
        log(f"    p_lower={r['p_lower']:.4f}  p_upper={r['p_upper']:.4f}  "
            f"p_TOST={r['p_tost']:.4f}  ->  {verdict} at alpha=0.05")

# ── SECONDARY: judge dimensions at S4 turn (supporting context only) ──────────
sub("SECONDARY — judge dimensions at S4 (overall, clarity, cogency, relevance)")
log("  [Supporting context only, not the primary resilience test]")
for dv in ["overall", "clarity", "cogency", "relevance"]:
    for cond in ["cfr_no_kg", "kg_cfr_full"]:
        g1 = df_s4_j.loc[df_s4_j["condition"] == cond, dv].values
        g2 = df_s4_j.loc[df_s4_j["condition"] == REF,  dv].values
        r  = tost(g1, g2, margin_sd=0.5, label=f"{dv}_S4: {cond} vs {REF}")
        tost_rows.append(r)
        verdict = "EQUIVALENT" if r["equiv"] else "not equivalent"
        log(f"\n  {r['label']}")
        if r.get("note") == "insufficient N":
            log(f"    n1={r['n1']}  n2={r['n2']}  *** INSUFFICIENT N ***")
        else:
            log(f"    n1={r['n1']}  n2={r['n2']}  "
                f"diff={r['diff']:+.4f}  pooled_sd={r['pooled_sd']:.4f}  "
                f"delta=+-{r['delta']:.4f}")
            log(f"    p_lower={r['p_lower']:.4f}  p_upper={r['p_upper']:.4f}  "
                f"p_TOST={r['p_tost']:.4f}  ->  {verdict} at alpha=0.05")

tost_df = pd.DataFrame(tost_rows)


# =============================================================================
#  EXPORT
# =============================================================================
section("EXPORT")

coef_df = pd.DataFrame(all_coef_rows)
coef_df.to_csv(COEF_F, index=False)
log(f"  Coefficient table -> {COEF_F}  ({len(coef_df)} rows)")

fdr_df.to_csv(FDR_F, index=False)
log(f"  FDR table         -> {FDR_F}  ({len(fdr_df)} rows)")

tost_df.to_csv(TOST_F, index=False)
log(f"  TOST table        -> {TOST_F}  ({len(tost_df)} rows)")

save_report()
log(f"  Full report       -> {REPORT_F}")
log("\nDone.")
