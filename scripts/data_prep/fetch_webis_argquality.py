"""
Fetch a reproducible N-sample subset of a human-annotated argument quality corpus
and save it in the format expected by run_e1_judge_calibration.py.

Sources tried in order:
  1. HuggingFace (several known dataset IDs)
  2. Zenodo -- Webis Argument Quality Corpus 2020 (DOI: 10.5281/zenodo.3780049)
     Gienapp et al. 2020, "Comparative Argument Quality Assessment: Crowdsourcing and Beyond"
     ACL 2020 -- https://zenodo.org/records/3780049
     1271 arguments, 20 topics, CC-BY-4.0

Output: data/webis_argquality_n<N>.json
  {"items": [{"text": "...", "context": "...", "label": <0-1 float>}, ...]}

Labels are normalised to [0, 1].  The E1 calibration script default
--label-normalization=auto will handle z-scored or 1-3 scale labels correctly.

Usage:
    python scripts/data_prep/fetch_webis_argquality.py --n 200 --seed 42
    python scripts/data_prep/fetch_webis_argquality.py --n 200 --local-csv data/webis-argquality-26k.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

# Webis Argument Quality Corpus 2020 -- Zenodo record 3780049
# Direct CSV download (1.3 MB, no unzipping needed)
_ZENODO_CSV_URL = (
    "https://zenodo.org/api/records/3780049/files/webis-argquality20-full.csv/content"
)
_WEBIS_QUALITY_COLS = [
    # Webis-ArgQuality-20 columns
    "overall_quality", "rhetorical_quality", "logical_quality", "dialectical_quality",
    # Webis-ArgQuality-26K columns (fallback)
    "overall", "cogency", "effectiveness", "reasonableness", "clarity",
]
_HF_CANDIDATES = [
    # ibm-research -- the real IBM ArgQuality 30k, available on HF
    # config required: 'argument_quality_ranking'
    ("ibm-research/argument_quality_ranking_30k", "argument_quality_ranking", "train",
     "argument", "topic", "WA"),
    # legacy IDs tried previously (kept for resilience)
    ("ibm/arg_quality_rank_30k",     None, "train", "argument", "topic", "WA"),
    ("ibm-arg-quality/arg_quality",  None, "train", "argument", "topic", "WA"),
    ("launch/arg_quality_rank_30k",  None, "train", "argument", "topic", "WA"),
    ("UKPLab/argument-aspect-rating",None, "train", "argument", "topic", "rating"),
    ("argsearch/ukp_aspect_corpus",  None, "train", "argument", "topic", "label"),
    ("argsearch/argument_quality",   None, "train", "argument", "topic", "label"),
]


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _pick_column(col_names: list, candidates: list) -> Optional[str]:
    for c in candidates:
        if c in col_names:
            return c
    return None


def _try_huggingface(n: int, seed: int) -> Optional[list]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("[fetch] 'datasets' not installed -- skipping HuggingFace sources.")
        return None

    for hf_path, hf_config, split, text_hint, ctx_hint, label_hint in _HF_CANDIDATES:
        info = f"{hf_path}" + (f" [{hf_config}]" if hf_config else "")
        print(f"[fetch] Trying HuggingFace: {info} ...")
        try:
            if hf_config:
                ds = load_dataset(hf_path, hf_config, split=split)
            else:
                ds = load_dataset(hf_path, split=split)
        except Exception as e:
            print(f"       Skipped: {str(e)[:100]}")
            continue

        cols = ds.column_names
        text_col  = _pick_column(cols, [text_hint,  "argument", "text", "claim",    "sentence"])
        ctx_col   = _pick_column(cols, [ctx_hint,   "topic",    "context", "prompt", "motion"])
        label_col = _pick_column(cols, [label_hint, "WA", "MACE-P", "rating",
                                        "quality", "label", "score"])

        if not text_col or not label_col:
            print(f"       Skipped: cannot identify text={text_col}/label={label_col}")
            continue

        print(f"       text='{text_col}' ctx='{ctx_col}' label='{label_col}' | {len(ds)} rows")
        rng = random.Random(seed)
        raw_rows = []
        for row in ds:
            text = str(row.get(text_col, "")).strip()
            ctx  = str(row.get(ctx_col, "")).strip() if ctx_col else ""
            try:
                label = float(row[label_col])
            except (TypeError, ValueError, KeyError):
                continue
            if text:
                raw_rows.append({"text": text, "context": ctx, "label": label})

        if not raw_rows:
            print("       Skipped: 0 valid items extracted")
            continue

        # Stratified sampling: 5 equal bands to ensure full quality spectrum
        labels_all = [r["label"] for r in raw_rows]
        lo_all, hi_all = min(labels_all), max(labels_all)
        span = hi_all - lo_all
        per_band = n // 5
        items: list = []
        for b in range(5):
            band_lo = lo_all + b * span / 5
            band_hi = lo_all + (b + 1) * span / 5 + (1e-9 if b == 4 else 0)
            pool = [r for r in raw_rows if band_lo <= r["label"] < band_hi]
            rng.shuffle(pool)
            items.extend(pool[:per_band])
        # top up if bands had fewer items
        remaining = n - len(items)
        if remaining > 0:
            used = set(id(r) for r in items)
            extra = [r for r in raw_rows if id(r) not in used]
            rng.shuffle(extra)
            items.extend(extra[:remaining])
        rng.shuffle(items)

        print(f"[fetch] SUCCESS via HuggingFace ({hf_path}): {len(items)} items")
        return items

    return None


def _fetch_webis_from_zenodo(n: int, seed: int) -> list:
    print("[fetch] Downloading Webis Argument Quality Corpus 2020 from Zenodo ...")
    print(f"        URL: {_ZENODO_CSV_URL}")
    print("        (approx 1.3 MB, direct CSV)")

    try:
        import requests as _req  # type: ignore
        resp = _req.get(_ZENODO_CSV_URL, allow_redirects=True, timeout=120,
                        headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        csv_text = resp.text
    except Exception as e:
        print(f"[fetch] ERROR downloading from Zenodo: {e}")
        print("\nManual fallback:")
        print("  1. Open: https://zenodo.org/records/3780049")
        print("  2. Download: webis-argquality20-full.csv")
        print("  3. Re-run with: --local-csv <path-to-csv>")
        sys.exit(1)

    print(f"[fetch] Download complete ({len(csv_text) // 1024} KB). Parsing ...")

    reader = csv.DictReader(io.StringIO(csv_text))
    fieldnames = reader.fieldnames or []
    print(f"[fetch] Columns: {fieldnames}")

    text_col  = _pick_column(fieldnames, ["argument", "text", "claim"])
    ctx_col   = _pick_column(fieldnames, ["topic", "context", "prompt"])
    label_col = _pick_column(fieldnames, _WEBIS_QUALITY_COLS)

    if not text_col or not label_col:
        print(f"[fetch] ERROR: cannot identify text={text_col}/label={label_col} in {fieldnames}")
        sys.exit(1)

    print(f"[fetch] Using text='{text_col}' ctx='{ctx_col}' label='{label_col}'")
    raw_rows = []
    for row in reader:
        text = str(row.get(text_col, "")).strip()
        ctx  = str(row.get(ctx_col, "")).strip() if ctx_col else ""
        try:
            label_raw = float(row[label_col])
        except (TypeError, ValueError, KeyError):
            continue
        if text:
            raw_rows.append({"text": text, "context": ctx, "label_raw": label_raw})

    if not raw_rows:
        print("[fetch] ERROR: No valid items extracted from CSV.")
        sys.exit(1)

    # normalise to [0, 1]
    labels_raw = [r["label_raw"] for r in raw_rows]
    lo, hi = min(labels_raw), max(labels_raw)
    span = hi - lo if hi != lo else 1.0
    for r in raw_rows:
        r["label"] = _clip01((r["label_raw"] - lo) / span)
        del r["label_raw"]

    rng = random.Random(seed)
    rng.shuffle(raw_rows)
    result = raw_rows[:n]
    if len(result) < n:
        print(f"[fetch] WARNING: only {len(result)} items available (requested {n})")
    print(f"[fetch] SUCCESS via Zenodo Webis-ArgQuality-20: {len(result)} items")
    return result


def _fetch_local_csv(path: Path, n: int, seed: int) -> list:
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        text_col  = _pick_column(fieldnames, ["argument", "text", "claim"])
        ctx_col   = _pick_column(fieldnames, ["topic", "context", "prompt"])
        label_col = _pick_column(fieldnames, _WEBIS_QUALITY_COLS + ["label", "score"])

        if not text_col or not label_col:
            print(f"ERROR: cannot find text/label columns in {fieldnames}")
            sys.exit(1)

        raw_rows = []
        for row in reader:
            text = str(row.get(text_col, "")).strip()
            ctx  = str(row.get(ctx_col,  "")).strip() if ctx_col else ""
            try:
                label = float(row[label_col])
            except (TypeError, ValueError):
                continue
            if text:
                raw_rows.append({"text": text, "context": ctx, "label_raw": label})

    labels_raw = [r["label_raw"] for r in raw_rows]
    lo, hi = min(labels_raw), max(labels_raw)
    span = hi - lo if hi != lo else 1.0
    for r in raw_rows:
        r["label"] = _clip01((r["label_raw"] - lo) / span)
        del r["label_raw"]

    rng = random.Random(seed)
    rng.shuffle(raw_rows)
    return raw_rows[:n]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch an argument quality corpus for E1 judge calibration"
    )
    parser.add_argument("--n", type=int, default=200,
                        help="Number of arguments to sample (default: 200)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output", default=None,
                        help="Output path (default: data/webis_argquality_n<N>.json)")
    parser.add_argument("--local-csv", default=None,
                        help="Path to a locally downloaded Webis ArgQuality CSV")
    args = parser.parse_args()

    out_path = (
        Path(args.output)
        if args.output
        else REPO_ROOT / "data" / f"webis_argquality_n{args.n}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.local_csv:
        items = _fetch_local_csv(Path(args.local_csv), args.n, args.seed)
        source = f"local:{args.local_csv}"
    else:
        items = _try_huggingface(args.n, args.seed)
        source = "huggingface"
        if items is None:
            items = _fetch_webis_from_zenodo(args.n, args.seed)
            source = f"zenodo:{_ZENODO_CSV_URL}"

    payload = {"items": items, "source": source, "n": len(items), "seed": args.seed}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    lo = min(r["label"] for r in items)
    hi = max(r["label"] for r in items)
    print(f"[fetch] Saved {len(items)} items to: {out_path}")
    print(f"[fetch] Label range: [{lo:.3f}, {hi:.3f}]")
    print()
    print("Run E1 calibration with:")
    print(
        f"  python scripts/analysis/run_e1_judge_calibration.py "
        f"--dataset {out_path} "
        f"--label-key label --text-key text --context-key context "
        f"--label-normalization auto "
        f"--min-pearson 0.3 --min-spearman 0.3 --max-mae 0.35"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
