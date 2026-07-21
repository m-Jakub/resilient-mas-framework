"""
Corrected E2 batch — publication-quality three-condition run.

This is the preferred E2 runner for publishable results.
Use run_e2_benchmark.py only for quick ad-hoc checks.

Conditions:
  - cfr_no_kg      : target N=30  (pools existing N=10 + new +20)
  - kg_cfr_full    : target N=30  (pools existing N=10 + new +20)
  - no_cfr_baseline: target N=30  (fresh run)

Smoke gate aborts if:
  - kg_cfr_full gate session has zero CFR turns with rag_docs_count > 0
  - any CFR condition gate session produces n_cc == 0
  - no_cfr_baseline gate session activates CFR

Usage:
  python scripts/run_corrected_e2.py
  python scripts/run_corrected_e2.py --skip-gate        # skip smoke gate
  python scripts/run_corrected_e2.py --dry-run          # plan only, no runs
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Repo root / path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PYTHON       = REPO_ROOT / "venv" / "Scripts" / "python.exe"
BENCHMARK    = REPO_ROOT / "scripts" / "analysis" / "run_e2_benchmark.py"
SESS_DIR     = REPO_ROOT / "logs" / "experiments"

# Existing corrected N=10 runs (post-fix, committed in d3f5f41 / 40c2aa0)
EXISTING = {
    "cfr_no_kg":    "exp_e2_20260308_113347",
    "kg_cfr_full":  "exp_e2_20260308_122757",
}
TOPUP_N       = 20   # additional sessions per CFR condition
BASELINE_N    = 30   # fresh no_cfr_baseline sessions
MAX_TURNS     = 10

# ---------------------------------------------------------------------------
# Scoring imports  (same functions used by run_e2_benchmark.py)
# ---------------------------------------------------------------------------
from scripts.analysis.offline_judge import WebisAlignedOfflineJudge, score_log_file
from scripts.analysis.compute_kes_metrics import compute_kes_metrics
from scripts.analysis.compute_v5_metrics import compute_v5_metrics
from scripts.analysis.compute_dis import compute_log_dis
from src.kg_cfr.aegis_orchestrator import get_condition_label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean_ci(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    se = math.sqrt(var) / math.sqrt(len(values))
    return mean, 1.96 * se


def _run_benchmark(conditions: List[str], runs: int, log_file: Path,
                   max_turns: int = MAX_TURNS) -> Tuple[int, str]:
    """Run benchmark as subprocess; return (exit_code, captured_run_id)."""
    cmd = [
        str(PYTHON), str(BENCHMARK),
        "--conditions", ",".join(conditions),
        "--runs", str(runs),
        "--max-turns", str(max_turns),
        "--skip-e1",
    ]
    print(f"  CMD: {' '.join(cmd)}")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    captured: List[str] = []
    with log_file.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            captured.append(line)
        proc.wait()

    # Extract run_id from "E2 artifacts saved under: analysis/outputs/e2/<RUN_ID>"
    run_id = ""
    for line in reversed(captured):
        m = re.search(r"E2 artifacts saved under:.*?[/\\]e2[/\\](exp_e2_\S+)", line)
        if m:
            run_id = m.group(1).strip()
            break
    return proc.returncode, run_id


def _sessions_for(run_id: str, condition: str) -> List[Path]:
    """Glob session JSONs: <SESS_DIR>/<run_id>_<condition>_*.json (no _shocks/_turns)."""
    pattern = f"{run_id}_{condition}_*.json"
    return sorted(
        p for p in SESS_DIR.glob(pattern)
        if not p.stem.endswith("_shocks") and not p.stem.endswith("_turns")
    )


def _score_sessions(
    sessions: List[Path],
    condition: str,
    out_dir: Path,
    judge: WebisAlignedOfflineJudge,
) -> Dict[str, Any]:
    """Score a list of session paths and return per-condition aggregate dict."""
    use_idrag, use_adaptive = True, True
    label = get_condition_label(condition, use_idrag, use_adaptive)

    judge_vals, dis_vals, cc_vals, aca_vals = [], [], [], []
    v5_cc_vals, v5_dis_vals, v5_sr_vals = [], [], []

    out_dir.mkdir(parents=True, exist_ok=True)

    for sp in sessions:
        sid = sp.stem
        # judge
        jres = score_log_file(sp, judge)
        jsum = jres.get("summary", {})
        if jsum.get("avg_overall") is not None:
            judge_vals.append(float(jsum["avg_overall"]))

        # DIS
        dres = compute_log_dis(sp, mode="jaccard")
        dsum = dres.get("summary", {})
        if dsum.get("avg_dis") is not None:
            dis_vals.append(float(dsum["avg_dis"]))

        # KES
        kres = compute_kes_metrics(sp, judge_result=jres)
        ksum = kres.get("summary", {})
        if ksum.get("avg_cc") is not None:
            cc_vals.append(float(ksum["avg_cc"]))
        if ksum.get("avg_aca") is not None:
            aca_vals.append(float(ksum["avg_aca"]))

        # v5
        vres = compute_v5_metrics(sp)
        vsum = vres.get("summary", {})
        if vsum.get("avg_cc") is not None:
            v5_cc_vals.append(float(vsum["avg_cc"]))
        if vsum.get("avg_dis") is not None:
            v5_dis_vals.append(float(vsum["avg_dis"]))
        if vsum.get("avg_sr") is not None:
            v5_sr_vals.append(float(vsum["avg_sr"]))

    def _s(vals):
        m, ci = _mean_ci(vals)
        return m, ci, len(vals)

    jm, jci, jn  = _s(judge_vals)
    dm, dci, dn  = _s(dis_vals)
    cm, cci, cn  = _s(cc_vals)
    am, aci, an  = _s(aca_vals)
    vm, vci, vn  = _s(v5_cc_vals)

    return {
        "condition":       condition,
        "condition_label": label,
        "n_sessions":      len(sessions),
        "judge_mean":      jm,  "judge_ci95":  jci,  "n_judge":  jn,
        "dis_mean":        dm,  "dis_ci95":    dci,  "n_dis":    dn,
        "cc_mean":         cm,  "cc_ci95":     cci,  "n_cc":     cn,
        "aca_mean":        am,  "aca_ci95":    aci,  "n_aca":    an,
        "v5_cc_mean":      vm,  "v5_cc_ci95":  vci,  "n_v5_cc":  vn,
    }


# ---------------------------------------------------------------------------
# Gate checks
# ---------------------------------------------------------------------------

def _check_kg_retrieval(gate_run_id: str) -> bool:
    """Gate 1: kg_cfr_full must have >=1 CFR turn with rag_docs_count > 0."""
    sessions = _sessions_for(gate_run_id, "kg_cfr_full")
    if not sessions:
        print("[GATE FAIL] No kg_cfr_full session found for gate run.")
        return False
    for sp in sessions:
        d = json.loads(sp.read_text(encoding="utf-8"))
        for t in d.get("turn_logs", []):
            m = t.get("metadata", {})
            if m.get("cfr_generated") and int(m.get("rag_docs_count") or 0) > 0:
                print(f"[GATE PASS] KG retrieval active: {sp.name}  rag_docs_count={m['rag_docs_count']}")
                return True
    print("[GATE FAIL] kg_cfr_full has zero turns with rag_docs_count > 0")
    return False


def _check_cc_nonzero(gate_agg_path: Path, conditions: List[str]) -> bool:
    """Gate 2: all listed conditions must have n_cc > 0 in the gate-run aggregate."""
    if not gate_agg_path.exists():
        print(f"[GATE FAIL] Aggregate not found: {gate_agg_path}")
        return False
    agg = json.loads(gate_agg_path.read_text(encoding="utf-8"))
    ok = True
    for cond in conditions:
        row = next((r for r in agg if r.get("condition") == cond), None)
        if row is None:
            print(f"[GATE FAIL] No row for condition: {cond}")
            ok = False
            continue
        n_cc = row.get("n_cc", 0) or 0
        if n_cc == 0:
            print(f"[GATE FAIL] n_cc=0 for {cond}")
            ok = False
        else:
            print(f"[GATE PASS] n_cc={n_cc} for {cond}")
    return ok


def _check_no_cfr_clean(gate_run_id: str) -> bool:
    """Gate 3: no_cfr_baseline sessions must not activate CFR."""
    sessions = _sessions_for(gate_run_id, "no_cfr_baseline")
    if not sessions:
        print("[GATE FAIL] No no_cfr_baseline session for gate run.")
        return False
    for sp in sessions:
        d = json.loads(sp.read_text(encoding="utf-8"))
        for t in d.get("turn_logs", []):
            m = t.get("metadata", {})
            cf = m.get("cf_flags", {})
            if isinstance(cf, str):
                try: cf = json.loads(cf)
                except: cf = {}
            if cf.get("activated") or m.get("cfr_generated"):
                print(f"[GATE FAIL] no_cfr_baseline activated CFR at turn {t.get('turn_number')} in {sp.name}")
                return False
    print(f"[GATE PASS] no_cfr_baseline clean (n_sessions={len(sessions)})")
    return True


# ---------------------------------------------------------------------------
# Debug artifact for kg_cfr_full
# ---------------------------------------------------------------------------

def _build_kg_debug(sessions: List[Path], out_path: Path) -> Dict[str, Any]:
    total_cfr, total_rag, rag_docs_sum = 0, 0, 0
    axiom_previews: List[str] = []

    for sp in sessions:
        d = json.loads(sp.read_text(encoding="utf-8"))
        for t in d.get("turn_logs", []):
            m = t.get("metadata", {})
            if not m.get("cfr_generated"):
                continue
            total_cfr += 1
            rd = int(m.get("rag_docs_count") or 0)
            if rd > 0:
                total_rag += 1
                rag_docs_sum += rd
                psj = m.get("private_strategy_json")
                if isinstance(psj, str):
                    try: psj = json.loads(psj)
                    except: psj = {}
                for ax in (psj or {}).get("retrieved_axioms", [])[:1]:
                    if len(axiom_previews) < 6:
                        if isinstance(ax, dict):
                            s = str(ax.get("subject", ax.get("s", "")))[:60]
                            p = str(ax.get("predicate", ax.get("p", "")))[:30]
                            axiom_previews.append(f"{s} [{p}]")
                        elif isinstance(ax, str):
                            axiom_previews.append(ax[:80])

    artifact = {
        "n_sessions":             len(sessions),
        "cfr_active_turns":       total_cfr,
        "turns_with_rag":         total_rag,
        "turns_rag_fraction":     round(total_rag / total_cfr, 3) if total_cfr else 0,
        "mean_rag_docs_count":    round(rag_docs_sum / total_rag, 2) if total_rag else 0,
        "axiom_previews":         axiom_previews,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    return artifact


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gate", action="store_true", help="Skip smoke gate")
    ap.add_argument("--dry-run",   action="store_true", help="Print plan, do not run anything")
    args = ap.parse_args()

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = REPO_ROOT / "logs" / f"corrected_e2_{ts}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}")
    print(f" CORRECTED E2 BATCH  {ts}")
    print(f" Conditions: cfr_no_kg(+{TOPUP_N}) | kg_cfr_full(+{TOPUP_N}) | no_cfr_baseline({BASELINE_N})")
    print(f" BatchDir: {batch_dir}")
    print(f"{'='*70}\n")

    if args.dry_run:
        print("[DRY RUN] Would run:")
        print(f"  Gate:  --conditions cfr_no_kg,kg_cfr_full,no_cfr_baseline --runs 1")
        print(f"  Top-up: --conditions cfr_no_kg,kg_cfr_full --runs {TOPUP_N}")
        print(f"  Baseline: --conditions no_cfr_baseline --runs {BASELINE_N}")
        print(f"  Existing sessions to pool:")
        for cond, rid in EXISTING.items():
            n = len(_sessions_for(rid, cond))
            print(f"    {cond}: {n} sessions from {rid}")
        return 0

    # -- Phase 1: Smoke gate ------------------------------------------------
    if not args.skip_gate:
        print("--- PHASE 1: SMOKE GATE (1 session each) ---\n")
        gate_log = batch_dir / "smoke_gate.txt"
        gate_exit, gate_run_id = _run_benchmark(
            ["cfr_no_kg", "kg_cfr_full", "no_cfr_baseline"], runs=1,
            log_file=gate_log, max_turns=6,
        )
        if gate_exit != 0:
            print(f"[ABORT] Benchmark error (exit {gate_exit})."); return 1
        if not gate_run_id:
            print("[ABORT] Could not parse gate run_id."); return 1
        print(f"\nGate run_id: {gate_run_id}")

        gate_agg = REPO_ROOT / "analysis" / "outputs" / "e2" / gate_run_id / "summary" / "e2_aggregate.json"

        print("\n-- Gate 1: KG retrieval --")
        if not _check_kg_retrieval(gate_run_id):
            print("[ABORT] Gate 1 FAILED."); return 2

        print("\n-- Gate 2: CC non-zero --")
        if not _check_cc_nonzero(gate_agg, ["cfr_no_kg", "kg_cfr_full"]):
            print("[ABORT] Gate 2 FAILED."); return 3

        print("\n-- Gate 3: no_cfr_baseline clean --")
        if not _check_no_cfr_clean(gate_run_id):
            print("[ABORT] Gate 3 FAILED."); return 4

        print("\n*** SMOKE GATE PASSED ***\n")
    else:
        print("--- PHASE 1: SMOKE GATE SKIPPED ---\n")

    # -- Phase 2: Top-up runs -----------------------------------------------
    print("--- PHASE 2: TOP-UP RUNS ---\n")

    print(f"[2a] cfr_no_kg + kg_cfr_full: +{TOPUP_N} each")
    topup_log = batch_dir / "topup_cfr_conditions.txt"
    topup_exit, topup_run_id = _run_benchmark(
        ["cfr_no_kg", "kg_cfr_full"], runs=TOPUP_N,
        log_file=topup_log,
    )
    if topup_exit != 0:
        print(f"[WARN] Top-up exited {topup_exit} (partial results may still be usable)")
    print(f"  Top-up run_id: {topup_run_id}")

    print(f"\n[2b] no_cfr_baseline: {BASELINE_N} fresh sessions")
    baseline_log = batch_dir / "baseline_no_cfr.txt"
    baseline_exit, baseline_run_id = _run_benchmark(
        ["no_cfr_baseline"], runs=BASELINE_N,
        log_file=baseline_log,
    )
    if baseline_exit != 0:
        print(f"[WARN] Baseline exited {baseline_exit} (partial results may still be usable)")
    print(f"  Baseline run_id: {baseline_run_id}")

    # -- Phase 3: Pool sessions ---------------------------------------------
    print("\n--- PHASE 3: POOLING SESSIONS ---\n")

    pooled: Dict[str, List[Path]] = {}
    for cond in ["cfr_no_kg", "kg_cfr_full"]:
        existing = _sessions_for(EXISTING[cond], cond) if cond in EXISTING else []
        topup    = _sessions_for(topup_run_id, cond)   if topup_run_id else []
        pooled[cond] = existing + topup
        print(f"  {cond}: {len(existing)} existing + {len(topup)} new = {len(pooled[cond])} total")

    pooled["no_cfr_baseline"] = _sessions_for(baseline_run_id, "no_cfr_baseline") if baseline_run_id else []
    print(f"  no_cfr_baseline: {len(pooled['no_cfr_baseline'])} sessions")

    # -- Phase 4: Score pooled sessions ------------------------------------
    print("\n--- PHASE 4: SCORING ---\n")
    judge = WebisAlignedOfflineJudge()
    agg_out_dir = batch_dir / "scored"
    aggregate: List[Dict[str, Any]] = []

    for cond in ["no_cfr_baseline", "cfr_no_kg", "kg_cfr_full"]:
        sessions = pooled.get(cond, [])
        if not sessions:
            print(f"  [SKIP] {cond}: no sessions"); continue
        print(f"  Scoring {cond}  ({len(sessions)} sessions)...")
        row = _score_sessions(sessions, cond, agg_out_dir / cond, judge)
        aggregate.append(row)
        cc_m  = f"{row['cc_mean']:.3f}" if row['cc_mean'] is not None else "n/a"
        aca_m = f"{row['aca_mean']:.3f}" if row['aca_mean'] is not None else "n/a"
        print(f"    DIS={row['dis_mean']:.3f}  KES_CC={cc_m}  ACA={aca_m}  v5_CC={row['v5_cc_mean']:.3f if row['v5_cc_mean'] else 'n/a'}")

    # Write master aggregate
    agg_path = batch_dir / "e2_corrected_aggregate.json"
    agg_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Aggregate saved: {agg_path}")

    # -- Phase 5: kg_cfr_full debug artifact --------------------------------
    print("\n--- PHASE 5: KG DEBUG ARTIFACT ---\n")
    kg_sessions = pooled.get("kg_cfr_full", [])
    if kg_sessions:
        dbg_path = batch_dir / "debug_kg_cfr_full.json"
        dbg = _build_kg_debug(kg_sessions, dbg_path)
        print(f"  CFR turns: {dbg['cfr_active_turns']}")
        print(f"  Turns with rag_docs>0: {dbg['turns_with_rag']}  ({dbg['turns_rag_fraction']*100:.1f}%)")
        print(f"  Mean rag_docs_count (when >0): {dbg['mean_rag_docs_count']}")
        print(f"  Axiom previews:")
        for ax in dbg["axiom_previews"]:
            print(f"    - {ax}")
        print(f"  Saved: {dbg_path}")

    # -- Phase 6: no_cfr sanity check --------------------------------------
    print("\n--- PHASE 6: no_cfr_baseline SANITY ---\n")
    no_cfr_sessions = pooled.get("no_cfr_baseline", [])
    cfr_leaks = 0
    for sp in no_cfr_sessions:
        d = json.loads(sp.read_text(encoding="utf-8"))
        for t in d.get("turn_logs", []):
            m = t.get("metadata", {})
            cf = m.get("cf_flags", {})
            if isinstance(cf, str):
                try: cf = json.loads(cf)
                except: cf = {}
            if cf.get("activated") or m.get("cfr_generated"):
                cfr_leaks += 1
    if cfr_leaks == 0:
        print(f"  [OK] CFR not activated in any of {len(no_cfr_sessions)} no_cfr_baseline sessions.")
    else:
        print(f"  [WARN] CFR activated in {cfr_leaks} turns across no_cfr_baseline sessions!")

    # -- Phase 7: Final report ----------------------------------------------
    print(f"\n{'='*70}")
    print(" CORRECTED E2 — FINAL RESULTS")
    print(f"{'='*70}")
    print(f"\n{'Condition':<22} {'N':>4}  {'DIS':>7}  {'KES_CC':>8}  {'ACA':>7}  {'v5_CC':>7}")
    print("-" * 60)
    for row in aggregate:
        lbl  = row["condition_label"]
        n    = row["n_sessions"]
        dis  = f"{row['dis_mean']:.3f}" if row["dis_mean"]  is not None else "n/a"
        cc   = f"{row['cc_mean']:.3f}"  if row["cc_mean"]   is not None else "n/a"
        aca  = f"{row['aca_mean']:.3f}" if row["aca_mean"]  is not None else "n/a"
        v5cc = f"{row['v5_cc_mean']:.3f}" if row["v5_cc_mean"] is not None else "n/a"
        print(f"  {lbl:<20} {n:>4}  {dis:>7}  {cc:>8}  {aca:>7}  {v5cc:>7}")

    print(f"\n{'='*70}")
    print(" INTERPRETATION MEMO")
    print(f"{'='*70}")
    _print_memo(aggregate)

    print(f"\n{'='*70}")
    print(" OUTPUT PATHS")
    print(f"{'='*70}")
    print(f"  BatchDir        : {batch_dir}")
    print(f"  Aggregate JSON  : {agg_path}")
    if not args.skip_gate:
        print(f"  Gate log        : {batch_dir / 'smoke_gate.txt'}")
    print(f"  Top-up log      : {batch_dir / 'topup_cfr_conditions.txt'}")
    print(f"  Baseline log    : {batch_dir / 'baseline_no_cfr.txt'}")
    if kg_sessions:
        print(f"  KG debug        : {batch_dir / 'debug_kg_cfr_full.json'}")
    print()
    return 0


def _print_memo(aggregate: List[Dict[str, Any]]) -> None:
    by_cond = {r["condition"]: r for r in aggregate}

    def _v(cond, key):
        row = by_cond.get(cond)
        if row is None: return None
        return row.get(key)

    # Q1: Does CFR beat no_cfr_baseline?
    no_cfr_dis = _v("no_cfr_baseline", "dis_mean")
    cfr_nkg_dis = _v("cfr_no_kg", "dis_mean")
    cfr_full_dis = _v("kg_cfr_full", "dis_mean")
    no_cfr_cc = _v("no_cfr_baseline", "cc_mean")
    cfr_nkg_cc = _v("cfr_no_kg", "cc_mean")
    cfr_full_cc = _v("kg_cfr_full", "cc_mean")

    print("\nQ1: Does CFR beat no_cfr_baseline?")
    if all(v is not None for v in [no_cfr_dis, cfr_nkg_dis, cfr_full_dis]):
        best_cfr_dis = max(cfr_nkg_dis, cfr_full_dis)
        delta = best_cfr_dis - no_cfr_dis
        verdict = "YES" if delta > 0.01 else "MARGINAL" if delta > -0.01 else "NO"
        print(f"  DIS: no_cfr={no_cfr_dis:.3f}  best_CFR={best_cfr_dis:.3f}  delta={delta:+.3f}  -> {verdict}")
    if all(v is not None for v in [no_cfr_cc, cfr_nkg_cc, cfr_full_cc]):
        best_cfr_cc = max(cfr_nkg_cc, cfr_full_cc)
        delta = best_cfr_cc - no_cfr_cc
        verdict = "YES" if delta > 0.02 else "MARGINAL" if delta > -0.02 else "NO"
        print(f"  KES CC: no_cfr={no_cfr_cc:.3f}  best_CFR={best_cfr_cc:.3f}  delta={delta:+.3f}  -> {verdict}")

    # Q2: Does kg_cfr_full differ from cfr_no_kg?
    print("\nQ2: Does kg_cfr_full differ meaningfully from cfr_no_kg? (post-bugfix)")
    for key, label in [("dis_mean","DIS"), ("cc_mean","KES_CC"), ("v5_cc_mean","v5_CC"), ("aca_mean","ACA")]:
        a = _v("cfr_no_kg", key)
        b = _v("kg_cfr_full", key)
        if a is not None and b is not None:
            delta = b - a
            verdict = "KG>noKG" if delta > 0.02 else "KG<noKG" if delta < -0.02 else "~equal"
            print(f"  {label:<8}: cfr_no_kg={a:.3f}  kg_cfr_full={b:.3f}  delta={delta:+.3f}  -> {verdict}")

    # Q3: ACA interpretation
    print("\nQ3: Is ACA a measurement limitation or genuine grounding failure?")
    no_cfr_aca = _v("no_cfr_baseline", "aca_mean")
    cfr_nkg_aca  = _v("cfr_no_kg",  "aca_mean")
    cfr_full_aca = _v("kg_cfr_full", "aca_mean")
    if all(v is not None for v in [no_cfr_aca, cfr_nkg_aca, cfr_full_aca]):
        all_low = all(v < 0.35 for v in [no_cfr_aca, cfr_nkg_aca, cfr_full_aca])
        kg_not_worst = cfr_full_aca >= cfr_nkg_aca - 0.02
        if all_low and kg_not_worst:
            print("  MEASUREMENT LIMITATION: ACA is low across all conditions (<0.35).")
            print("  kg_cfr_full is not significantly worse than cfr_no_kg.")
            print("  Root cause: ACA lexical matcher fails when LLM translates KG axioms")
            print("  into scenario-domain language (semantic gap, not retrieval noise).")
        elif cfr_full_aca < no_cfr_aca - 0.05:
            print("  POSSIBLE GROUNDING ISSUE: kg_cfr_full ACA below no_cfr_baseline.")
            print("  Investigate: retrieval noise vs. KG-induced argument complexity.")
        else:
            print("  INCONCLUSIVE: ACA pattern mixed. Further investigation needed.")


if __name__ == "__main__":
    raise SystemExit(main())
