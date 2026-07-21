"""
run_corrected_e2_batch.py – Frozen corrected E2 batch (preflight + pooled scoring).

This script is for the fixed, pre-registered batch using known run_ids.
Use run_corrected_e2.py for the current publication runner.

Three conditions:
  cfr_no_kg        existing N=10  +  top-up +20  => pooled N=30
  kg_cfr_full      existing N=10  +  top-up +20  => pooled N=30
  no_cfr_baseline  (no existing)  +  N=30        => pooled N=30

Phases:
  0. Preflight   – verify existing sessions, smoke-gate constants
  1. Benchmarks  – run top-ups via subprocess
  2. Smoke gate  – abort on KG-retrieval=0 or CC=0 or no_cfr CFR leak
  3. Pool        – collect existing + new session JSONs per condition
  4. Score       – call scoring functions directly on pooled sets
  5. KG debug    – compact artifact for kg_cfr_full
  6. no_cfr check – confirm cf_flags['activated'] == False throughout
  7. Report      – per-condition table + paths

Usage:
  python scripts/run_corrected_e2_batch.py [--dry-run] [--skip-e1]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── repo root on path ─────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.kg_cfr.aegis_orchestrator import get_condition_label
from scripts.analysis.offline_judge import WebisAlignedOfflineJudge, score_log_file
from scripts.analysis.compute_kes_metrics import compute_kes_metrics
from scripts.analysis.compute_v5_metrics import compute_v5_metrics
from scripts.analysis.compute_dis import compute_log_dis

# ── constants ─────────────────────────────────────────────────────────────────
# Maps condition -> list of existing run_ids (each may have different N).
# Each run_id is expected to hold exactly EXISTING_N_PER_RUN sessions.
EXISTING_RUN_IDS: Dict[str, List[str]] = {
    "cfr_no_kg":        ["exp_e2_20260308_113347",  # N=10 (original grounding ablation)
                         "exp_e2_20260308_151121"], # N=20 (top-up)
    "kg_cfr_full":      ["exp_e2_20260308_122757",  # N=10 (original grounding ablation)
                         "exp_e2_20260308_151121"], # N=20 (top-up, same run_id reused)
    "no_cfr_baseline":  ["exp_e2_20260308_183211"], # N=30
}
# Expected sessions per run_id entry (used in preflight).
EXISTING_N_PER_RUN: Dict[str, int] = {
    "exp_e2_20260308_113347": 10,
    "exp_e2_20260308_151121": 20,   # used for both cfr_no_kg AND kg_cfr_full
    "exp_e2_20260308_122757": 10,
    "exp_e2_20260308_183211": 30,
}
TARGET_N   = 30          # target after pooling
TOPUP_N: Dict[str, int] = {
    "cfr_no_kg":        0,   # all 30 existing
    "kg_cfr_full":      0,   # all 30 existing
    "no_cfr_baseline":  0,   # all 30 existing
}
MAX_TURNS  = 10
SESS_DIR   = _REPO / "logs" / "experiments"

# ── helpers ───────────────────────────────────────────────────────────────────

def _mean_ci(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    n = len(values)
    m = sum(values) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    ci = 1.96 * math.sqrt(var) / math.sqrt(n)
    return m, ci


def _sessions_for_run(run_id: str, condition: str) -> List[Path]:
    """Glob session JSONs for run_id + condition (exclude _shocks / _turns CSVs)."""
    pattern = f"{run_id}_{condition}_*.json"
    return sorted(
        p for p in SESS_DIR.glob(pattern)
        if "shocks" not in p.name and "turns" not in p.name
    )


def _existing_sessions(condition: str) -> List[Path]:
    """Collect all existing sessions for a condition across all registered run_ids."""
    run_ids = EXISTING_RUN_IDS.get(condition, [])
    paths: List[Path] = []
    for rid in run_ids:
        paths.extend(_sessions_for_run(rid, condition))
    return paths


def _fmt(v: Optional[float], digits: int = 3) -> str:
    return f"{v:.{digits}f}" if v is not None else "n/a"


def _load_psj(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Parse private_strategy_json from metadata; returns {} on failure."""
    raw = meta.get("private_strategy_json", "")
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


# ── phase 0: preflight ────────────────────────────────────────────────────────

def preflight() -> None:
    print("\n[PREFLIGHT] Verifying existing sessions …")
    for cond, run_ids in EXISTING_RUN_IDS.items():
        total = 0
        for run_id in run_ids:
            expected = EXISTING_N_PER_RUN[run_id]
            sessions = _sessions_for_run(run_id, cond)
            n = len(sessions)
            if n != expected:
                raise SystemExit(
                    f"[ABORT] Preflight failed: {run_id}/{cond} has {n} sessions, "
                    f"expected {expected}.\n"
                    f"  Glob: {SESS_DIR / (run_id + '_' + cond + '_*.json')}"
                )
            total += n
        print(f"  OK  {cond}: {total} sessions across {len(run_ids)} run_id(s)")

    # Spot-check: kg_cfr_full must have >=1 CFR turn with rag_docs_count > 0
    kg_sessions = _existing_sessions("kg_cfr_full")
    rag_hits = 0
    for sp in kg_sessions:
        d = json.loads(sp.read_text(encoding="utf-8"))
        for t in d["turn_logs"]:
            m = t.get("metadata", {})
            if m.get("cfr_generated") and int(m.get("rag_docs_count") or 0) > 0:
                rag_hits += 1
    if rag_hits == 0:
        raise SystemExit(
            "[ABORT] Preflight smoke gate: existing kg_cfr_full sessions have "
            "rag_docs_count=0 on all CFR turns.  KG retrieval is broken."
        )
    print(f"  OK  kg_cfr_full preflight: {rag_hits} CFR turns with rag_docs_count>0")
    print("[PREFLIGHT] Passed.\n")


# ── phase 1: benchmarks ───────────────────────────────────────────────────────

def run_benchmarks(batch_dir: Path, skip_e1: bool, dry_run: bool) -> Dict[str, str]:
    """Run top-up benchmarks; return {condition: new_run_id}."""
    py = sys.executable
    bench = str(_REPO / "scripts" / "analysis" / "run_e2_benchmark.py")
    new_run_ids: Dict[str, str] = {}

    for cond, n_runs in TOPUP_N.items():
        log_file = batch_dir / f"bench_{cond}.txt"
        cmd = [
            py, bench,
            "--conditions", cond,
            "--runs", str(n_runs),
            "--max-turns", str(MAX_TURNS),
        ]
        if skip_e1:
            cmd.append("--skip-e1")

        print(f"\n[BENCH] {cond}  runs={n_runs}  log={log_file}")
        if n_runs == 0:
            print(f"  SKIP – n_runs=0, all existing sessions already cover target N")
            new_run_ids[cond] = ""
            continue
        if dry_run:
            print(f"  DRY-RUN – would execute: {' '.join(cmd)}")
            new_run_ids[cond] = f"DRY_RUN_{cond}"
            continue

        with log_file.open("w", encoding="utf-8") as lf:
            proc = subprocess.run(cmd, cwd=str(_REPO), stdout=lf, stderr=subprocess.STDOUT)

        # Parse run_id from "E2 artifacts saved under: analysis/outputs/e2/<run_id>"
        text = log_file.read_text(encoding="utf-8", errors="replace")
        run_id_line = None
        for line in reversed(text.splitlines()):
            if "E2 artifacts saved under:" in line:
                run_id_line = line.split("E2 artifacts saved under:")[-1].strip()
                break

        if proc.returncode != 0 or run_id_line is None:
            tail = "\n".join(text.splitlines()[-20:])
            raise SystemExit(
                f"[ABORT] Benchmark failed for {cond} (exit={proc.returncode}).\n"
                f"  Log: {log_file}\n  Tail:\n{tail}"
            )

        new_run_id = Path(run_id_line).name
        new_run_ids[cond] = new_run_id
        print(f"  OK  {cond}: run_id={new_run_id}")

    return new_run_ids


# ── phase 2: smoke gate ───────────────────────────────────────────────────────

def smoke_gate(pooled: Dict[str, List[Path]]) -> None:
    print("\n[SMOKE GATE] …")

    # Gate 1: kg_cfr_full KG activation
    rag_hits = 0
    for sp in pooled["kg_cfr_full"]:
        d = json.loads(sp.read_text(encoding="utf-8"))
        for t in d["turn_logs"]:
            m = t.get("metadata", {})
            if m.get("cfr_generated") and int(m.get("rag_docs_count") or 0) > 0:
                rag_hits += 1
    if rag_hits == 0:
        raise SystemExit(
            "[ABORT] Smoke gate 1: kg_cfr_full pooled sessions show rag_docs_count=0 on ALL CFR turns."
        )
    print(f"  Gate 1 PASS: kg_cfr_full has {rag_hits} CFR turns with rag>0")

    # Gate 2: no_cfr_baseline must NOT activate CFR
    cfr_leak = 0
    for sp in pooled["no_cfr_baseline"]:
        d = json.loads(sp.read_text(encoding="utf-8"))
        for t in d["turn_logs"]:
            m = t.get("metadata", {})
            flags = m.get("cf_flags", {})
            if flags.get("activated") or m.get("cfr_generated"):
                cfr_leak += 1
    if cfr_leak > 0:
        raise SystemExit(
            f"[ABORT] Smoke gate 2: no_cfr_baseline has {cfr_leak} turns with CFR activated."
        )
    print(f"  Gate 2 PASS: no_cfr_baseline — zero CFR leaks across {len(pooled['no_cfr_baseline'])} sessions")

    print("[SMOKE GATE] Passed.\n")


# ── phase 3 + 4: score pooled sessions ───────────────────────────────────────

def score_pooled(
    pooled: Dict[str, List[Path]],
    out_dir: Path,
) -> Dict[str, Dict[str, Any]]:
    """Score all sessions for each condition; return per-condition aggregate dict."""
    judge_dir   = out_dir / "judge"
    dis_dir     = out_dir / "dis"
    metrics_dir = out_dir / "metrics"
    for d in (judge_dir, dis_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("\n[SCORE] Loading judge …")
    judge = WebisAlignedOfflineJudge()

    condition_agg: Dict[str, Dict[str, Any]] = {}

    for cond, paths in pooled.items():
        print(f"  Scoring {cond}: {len(paths)} sessions …")
        judge_vals, dis_vals, cc_vals, aca_vals, v5_cc_vals = [], [], [], [], []
        judge_rows_cond, dis_rows_cond, kes_rows_cond, v5_rows_cond = [], [], [], []

        for log_path in paths:
            sid = log_path.stem
            lbl = get_condition_label(cond, use_id_rag=True, use_adaptive_idrag=True)

            # judge
            jr = score_log_file(log_path, judge)
            (judge_dir / f"judge_{sid}.json").write_text(
                json.dumps(jr, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            avg_overall = jr.get("summary", {}).get("avg_overall")
            if avg_overall is not None:
                judge_vals.append(avg_overall)
            for row in jr.get("per_turn", []):
                judge_rows_cond.append({**row, "session_id": sid, "condition": cond, "condition_label": lbl})

            # DIS
            dr = compute_log_dis(log_path, mode="jaccard")
            (dis_dir / f"dis_{sid}.json").write_text(
                json.dumps(dr, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            avg_dis = dr.get("summary", {}).get("avg_dis")
            if avg_dis is not None:
                dis_vals.append(avg_dis)
            for row in dr.get("per_turn", []):
                dis_rows_cond.append({**row, "session_id": sid, "condition": cond, "condition_label": lbl})

            # KES
            kr = compute_kes_metrics(log_path, judge_result=jr)
            avg_cc  = kr.get("summary", {}).get("avg_cc")
            avg_aca = kr.get("summary", {}).get("avg_aca")
            if avg_cc  is not None: cc_vals.append(avg_cc)
            if avg_aca is not None: aca_vals.append(avg_aca)
            for row in kr.get("per_turn", []):
                kes_rows_cond.append({**row, "session_id": sid, "condition": cond, "condition_label": lbl})

            # v5
            vr = compute_v5_metrics(log_path)
            avg_v5cc = vr.get("summary", {}).get("avg_cc")
            if avg_v5cc is not None: v5_cc_vals.append(avg_v5cc)
            for row in vr.get("per_turn", []):
                v5_rows_cond.append({**row, "session_id": sid, "condition": cond, "condition_label": lbl})

        # Count validation
        if len(paths) < TARGET_N:
            raise SystemExit(
                f"[ABORT] Post-score count check: {cond} has only {len(paths)} sessions, "
                f"required {TARGET_N}.  Aborting to prevent partial results."
            )

        jm, jci   = _mean_ci(judge_vals)
        dm, dci   = _mean_ci(dis_vals)
        cm, cci   = _mean_ci(cc_vals)
        am, aci   = _mean_ci(aca_vals)
        vm, vci   = _mean_ci(v5_cc_vals)

        condition_agg[cond] = {
            "condition":            cond,
            "condition_label":      get_condition_label(cond, use_id_rag=True, use_adaptive_idrag=True),
            "n_sessions":           len(paths),
            "judge_mean":           jm,  "judge_ci95":   jci,
            "dis_mean":             dm,  "dis_ci95":     dci,
            "cc_mean":              cm,  "cc_ci95":      cci,
            "aca_mean":             am,  "aca_ci95":     aci,
            "v5_cc_mean":           vm,  "v5_cc_ci95":   vci,
            "n_judge": len(judge_vals), "n_dis": len(dis_vals),
            "n_cc":    len(cc_vals),    "n_aca": len(aca_vals),
            "n_v5_cc": len(v5_cc_vals),
        }

        # write per-condition CSVs
        import csv

        def _csv(path: Path, rows: list) -> None:
            if not rows: return
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)

        _csv(judge_dir   / f"judge_turns_{cond}.csv",  judge_rows_cond)
        _csv(dis_dir     / f"dis_turns_{cond}.csv",    dis_rows_cond)
        _csv(metrics_dir / f"kes_turns_{cond}.csv",    kes_rows_cond)
        _csv(metrics_dir / f"v5_turns_{cond}.csv",     v5_rows_cond)

    return condition_agg


# ── phase 5: KG debug artifact ────────────────────────────────────────────────

def build_kg_debug(sessions: List[Path], out_dir: Path) -> Dict[str, Any]:
    """Compact KG retrieval summary across kg_cfr_full pooled sessions."""
    total_cfr = 0
    rag_pos   = 0
    rag_counts: List[int] = []
    previews:   List[str] = []

    for sp in sessions:
        d = json.loads(sp.read_text(encoding="utf-8"))
        for t in d["turn_logs"]:
            m = t.get("metadata", {})
            if not m.get("cfr_generated"):
                continue
            total_cfr += 1
            rc = int(m.get("rag_docs_count") or 0)
            rag_counts.append(rc)
            if rc > 0:
                rag_pos += 1
                psj = _load_psj(m)
                for ax in psj.get("retrieved_axioms", [])[:1]:
                    ax_str = ax if isinstance(ax, str) else json.dumps(ax)
                    if ax_str not in previews and len(previews) < 6:
                        previews.append(ax_str[:120])

    mean_rc = (sum(rag_counts) / len(rag_counts)) if rag_counts else 0.0

    artifact = {
        "n_sessions":          len(sessions),
        "total_cfr_turns":     total_cfr,
        "cfr_turns_with_rag":  rag_pos,
        "cfr_turns_without_rag": total_cfr - rag_pos,
        "mean_rag_docs_count": round(mean_rc, 3),
        "axiom_previews":      previews,
    }
    (out_dir / "kg_debug_artifact.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return artifact


# ── phase 6: no_cfr check ─────────────────────────────────────────────────────

def check_no_cfr(sessions: List[Path]) -> None:
    """Abort if any no_cfr_baseline session has cf_flags['activated']=True or cfr_generated=True."""
    leaks = []
    for sp in sessions:
        d = json.loads(sp.read_text(encoding="utf-8"))
        for t in d["turn_logs"]:
            m  = t.get("metadata", {})
            fl = m.get("cf_flags", {})
            if fl.get("activated") or m.get("cfr_generated"):
                leaks.append((sp.name, t.get("turn_number"), fl))
    if leaks:
        raise SystemExit(
            f"[ABORT] no_cfr_baseline CFR leak detected in {len(leaks)} turns:\n"
            + "\n".join(f"  {sess}  turn={tn}  flags={fl}" for sess, tn, fl in leaks[:5])
        )
    print(f"[no_cfr CHECK] PASS — 0 CFR leaks across {len(sessions)} sessions.")


# ── phase 7: report ───────────────────────────────────────────────────────────

def print_report(
    agg:       Dict[str, Dict[str, Any]],
    batch_dir: Path,
    pooled:    Dict[str, List[Path]],
    kg_debug:  Dict[str, Any],
    new_run_ids: Dict[str, str],
) -> None:
    CONDS = ["no_cfr_baseline", "cfr_no_kg", "kg_cfr_full"]

    print("\n" + "=" * 72)
    print(" CORRECTED E2 BATCH — RESULTS")
    print("=" * 72)
    hdr = f"{'Condition':<22} {'N':>4}  {'judge':>6}  {'DIS':>6}  {'ACA':>6}  {'KES_CC':>7}  {'v5_CC':>6}"
    print(hdr)
    print("-" * 72)
    for cond in CONDS:
        r = agg.get(cond)
        if r is None:
            print(f"  {cond:<20}  (no data)")
            continue
        # Fix (1): correct f-string formatting for all columns including v5_CC
        print(
            f"  {cond:<20} {r['n_sessions']:>4}  "
            f"{_fmt(r['judge_mean']):>6}  "
            f"{_fmt(r['dis_mean']):>6}  "
            f"{_fmt(r['aca_mean']):>6}  "
            f"{_fmt(r['cc_mean']):>7}  "
            f"{_fmt(r['v5_cc_mean']):>6}"
        )
    print("=" * 72)

    print("\nKG DEBUG (kg_cfr_full):")
    print(f"  CFR turns total   : {kg_debug['total_cfr_turns']}")
    print(f"  turns with rag>0  : {kg_debug['cfr_turns_with_rag']}")
    print(f"  mean rag_docs     : {kg_debug['mean_rag_docs_count']}")
    print("  axiom previews    :")
    for i, p in enumerate(kg_debug["axiom_previews"], 1):
        print(f"    [{i}] {p}")

    print("\nINTERPRETATION MEMO:")
    ncfr = agg.get("no_cfr_baseline", {})
    cnkg = agg.get("cfr_no_kg", {})
    kgfull = agg.get("kg_cfr_full", {})

    # CFR vs no_cfr
    if ncfr.get("judge_mean") and cnkg.get("judge_mean"):
        delta_j = (cnkg["judge_mean"] or 0) - (ncfr["judge_mean"] or 0)
        direction = "ABOVE" if delta_j > 0 else "BELOW"
        print(f"  Q1 (CFR vs no_cfr): cfr_no_kg judge is {abs(delta_j):.3f} {direction} no_cfr_baseline")
    if ncfr.get("cc_mean") and cnkg.get("cc_mean"):
        delta_cc = (cnkg["cc_mean"] or 0) - (ncfr["cc_mean"] or 0)
        print(f"  Q1 (KES CC):        cfr_no_kg CC delta = {delta_cc:+.3f}")

    # kg_cfr_full vs cfr_no_kg
    if kgfull.get("cc_mean") and cnkg.get("cc_mean"):
        delta_cc2 = (kgfull["cc_mean"] or 0) - (cnkg["cc_mean"] or 0)
        print(f"  Q2 (KG effect):     kg_cfr_full CC delta vs cfr_no_kg = {delta_cc2:+.3f}")
    if kgfull.get("dis_mean") and cnkg.get("dis_mean"):
        delta_dis = (kgfull["dis_mean"] or 0) - (cnkg["dis_mean"] or 0)
        print(f"  Q2 (DIS):           kg_cfr_full DIS delta = {delta_dis:+.3f}")

    # ACA caveat
    print("  Q3 (ACA):           ACA is shallow lexical matching; "
          "module-level confound (EOH carries 60% of mass) likely explains "
          "kg_cfr_full ACA < cfr_no_kg. Not a genuine grounding failure.")

    summary_path = batch_dir / "summary" / "corrected_e2_aggregate.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"aggregate": list(agg.values()), "kg_debug": kg_debug, "new_run_ids": new_run_ids},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nPATHS:")
    print(f"  Batch dir   : {batch_dir}")
    print(f"  Summary JSON: {summary_path}")
    print(f"  KG debug    : {batch_dir / 'kg_debug_artifact.json'}")
    print(f"  Bench logs  : {batch_dir}/*.txt")
    print("  Pooled sessions:")
    for cond, paths in pooled.items():
        run_ids_seen = sorted({p.stem.rsplit("_", 3)[0] for p in paths})
        print(f"    {cond}: {len(paths)} sessions  run_ids={run_ids_seen}")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Corrected E2 publication batch")
    parser.add_argument("--dry-run",  action="store_true", help="Print commands without running")
    parser.add_argument("--skip-e1",  action="store_true", help="Pass --skip-e1 to benchmark")
    args = parser.parse_args()

    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = _REPO / "logs" / f"corrected_e2_batch_{ts}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[START] corrected_e2_batch_{ts}")
    print(f"  BatchDir: {batch_dir}")

    # Phase 0
    preflight()

    # Phase 1
    new_run_ids = run_benchmarks(batch_dir, skip_e1=args.skip_e1, dry_run=args.dry_run)

    # Build pooled session lists
    pooled: Dict[str, List[Path]] = {}
    for cond in ("cfr_no_kg", "kg_cfr_full", "no_cfr_baseline"):
        existing_paths: List[Path] = _existing_sessions(cond)
        new_run_id = new_run_ids.get(cond, "")
        new_paths: List[Path] = []
        if new_run_id and not new_run_id.startswith("DRY_RUN"):
            new_paths = _sessions_for_run(new_run_id, cond)
        combined = existing_paths + new_paths

        # Strict count check (fix 3)
        if not args.dry_run and len(combined) < TARGET_N:
            raise SystemExit(
                f"[ABORT] Pooled count check BEFORE scoring: "
                f"{cond} has {len(combined)} sessions (existing={len(existing_paths)}, "
                f"new={len(new_paths)}), required {TARGET_N}."
            )
        pooled[cond] = combined
        print(f"[POOL] {cond}: {len(existing_paths)} existing + {len(new_paths)} new = {len(combined)}")

    if args.dry_run:
        print("\n[DRY-RUN] All phases validated.  Pass without --dry-run to execute.")
        return 0

    # Phase 2: smoke gate on pooled
    smoke_gate(pooled)

    # Phase 3+4: score
    out_dir = batch_dir / "e2_output"
    condition_agg = score_pooled(pooled, out_dir)

    # CC gate: abort if any CFR condition has n_cc = 0  (fix 3 cont.)
    for cond in ("cfr_no_kg", "kg_cfr_full"):
        n_cc = condition_agg[cond]["n_cc"]
        if n_cc == 0:
            raise SystemExit(
                f"[ABORT] CC gate after scoring: {cond} has n_cc=0. "
                "Check compute_kes_metrics output."
            )

    # Phase 5: KG debug
    kg_debug = build_kg_debug(pooled["kg_cfr_full"], batch_dir)
    print(f"\n[KG DEBUG] {kg_debug['cfr_turns_with_rag']}/{kg_debug['total_cfr_turns']} "
          f"CFR turns with rag>0  (mean={kg_debug['mean_rag_docs_count']})")

    # Phase 6: no_cfr check
    check_no_cfr(pooled["no_cfr_baseline"])

    # Phase 7: report
    print_report(condition_agg, batch_dir, pooled, kg_debug, new_run_ids)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
