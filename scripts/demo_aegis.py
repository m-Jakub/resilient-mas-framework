"""
Live console demo for the AEGIS/DRAU Tripartite Standoff.

Runs a single session and prints turns with ANSI colors for qualitative analysis.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root and src/ are on sys.path when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyYAML is required for demo_aegis.py. Install with: pip install pyyaml"
    ) from exc

from src.kg_cfr.aegis_logger import AegisLogger
from src.kg_cfr.aegis_orchestrator import TripartiteStandoff, create_default_modules
from config.settings import DILEMMAS_AEGIS_FILE


ANSI_RESET = "\x1b[0m"
ANSI_DIM = "\x1b[2m"
ANSI_BLUE = "\x1b[34m"
ANSI_GREEN = "\x1b[32m"
ANSI_YELLOW = "\x1b[33m"
ANSI_RED_BOLD = "\x1b[1;31m"


def _load_dilemmas(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dilemmas file: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data.get("dilemmas", []) if isinstance(data, dict) else []


def _pick_dilemma(dilemmas: List[Dict[str, Any]], dilemma_id: Optional[str]) -> Dict[str, Any]:
    if not dilemmas:
        raise ValueError("No dilemmas found in dilemmas_aegis.yml")
    if dilemma_id:
        for d in dilemmas:
            if d.get("id") == dilemma_id:
                return d
        raise ValueError(f"Dilemma id not found: {dilemma_id}")
    return dilemmas[0]


def _wrap(text: str, width: int = 96) -> str:
    return "\n".join(textwrap.fill(line, width=width) for line in text.splitlines())


class LiveAegisLogger(AegisLogger):
    def __init__(
        self,
        session_id: str,
        crisis_context: str,
        color_map: Dict[str, str],
        sleep_s: float,
        transcript_path: Optional[Path] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(session_id, crisis_context, metadata=metadata)
        self.color_map = color_map
        self.sleep_s = sleep_s
        self.transcript_path = transcript_path
        self.transcript_lines: List[str] = []

    def log_shock(
        self,
        shock_type: str,
        shock_text: str,
        turn_number: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().log_shock(shock_type, shock_text, turn_number, metadata=metadata)
        header = f"[SHOCK {shock_type}] Turn {turn_number}"
        print(f"{ANSI_RED_BOLD}{header}{ANSI_RESET}")
        print(f"{ANSI_RED_BOLD}{_wrap(shock_text)}{ANSI_RESET}")
        self.transcript_lines.append(header)
        self.transcript_lines.append(_wrap(shock_text))
        time.sleep(self.sleep_s)

    def log_turn(
        self,
        turn_number: int,
        phase: str,
        module_id: str,
        module_label: str,
        message_content: str,
        concepts_mentioned: Optional[List[str]] = None,
        response_time_ms: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().log_turn(
            turn_number=turn_number,
            phase=phase,
            module_id=module_id,
            module_label=module_label,
            message_content=message_content,
            concepts_mentioned=concepts_mentioned,
            response_time_ms=response_time_ms,
            metadata=metadata,
        )

        color = self.color_map.get(module_id, "")
        header = f"[Turn {turn_number}] {module_label}"

        cfr_tag = ""
        cf_flags = (metadata or {}).get("cf_flags", {})
        if cf_flags.get("activated"):
            if cf_flags.get("cfr_mode") in ("kg_cfr_full", "knowledge-grounded"):
                cfr_tag = "[KG-CFR Activated]"
            else:
                cfr_tag = "[CFR Activated]"

        if cfr_tag:
            header = f"{header} {ANSI_DIM}{cfr_tag}{ANSI_RESET}"

        print(f"{color}{header}{ANSI_RESET}")
        print(f"{color}{_wrap(message_content)}{ANSI_RESET}")
        header_plain = f"[Turn {turn_number}] {module_label}"
        if cfr_tag:
            header_plain = f"{header_plain} {cfr_tag}"
        self.transcript_lines.append(header_plain)
        self.transcript_lines.append(_wrap(message_content))
        time.sleep(self.sleep_s)

    def save_transcript(self) -> None:
        if not self.transcript_path:
            return
        self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        self.transcript_path.write_text("\n".join(self.transcript_lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Live AEGIS/DRAU demo (single session).")
    parser.add_argument("--dilemma-id", type=str, default=None, help="Dilemma id to run")
    parser.add_argument("--max-turns", type=int, default=6, help="Number of turns (default: 6)")
    parser.add_argument(
        "--cfr-mode",
        type=str,
        default="kg_cfr_full",
        choices=[
            "no_cfr_baseline", "cfr_no_kg", "kg_cfr_full",          # canonical
            "none", "prompt-only", "knowledge-grounded",            # legacy aliases
        ],
        help=(
            "CFR mode for the demo (default: kg_cfr_full). "
            "Canonical: no_cfr_baseline | cfr_no_kg | kg_cfr_full. "
            "Legacy aliases: none | prompt-only | knowledge-grounded."
        ),
    )
    parser.add_argument("--sleep", type=float, default=2.0, help="Delay between turns (seconds)")

    args = parser.parse_args()

    dilemmas = _load_dilemmas(DILEMMAS_AEGIS_FILE)
    dilemma = _pick_dilemma(dilemmas, args.dilemma_id)

    crisis_context = {
        "id": dilemma.get("id"),
        "title": dilemma.get("title"),
        "prompt": dilemma.get("prompt"),
    }

    color_map = {
        "eoh": ANSI_BLUE,
        "ivc": ANSI_GREEN,
        "shm": ANSI_YELLOW,
    }

    standoff = TripartiteStandoff(
        debate_turns=args.max_turns,
        enable_shocks=True,
        shock_type="auto",
        cfr_mode=args.cfr_mode,
        dilemma_id=crisis_context.get("id"),
        enable_logging=True,
    )

    standoff.logger = LiveAegisLogger(
        session_id=standoff.session_id,
        crisis_context=str(crisis_context),
        color_map=color_map,
        sleep_s=args.sleep,
        transcript_path=Path("logs/experiments") / f"{standoff.session_id}_demo.txt",
        metadata={
            "debate_turns": args.max_turns,
            "shock_type": "auto",
            "cfr_mode": args.cfr_mode,
            "dilemma_id": crisis_context.get("id"),
        },
    )

    print("=" * 80)
    print("AEGIS/DRAU LIVE DEMO")
    print("=" * 80)
    print(f"Dilemma: {crisis_context['title']}")
    print(f"CFR mode: {args.cfr_mode}")
    print(f"Turns: {args.max_turns}")
    print("=" * 80)
    print(_wrap(crisis_context["prompt"]))
    print("=" * 80)
    time.sleep(args.sleep)

    modules = create_default_modules()
    standoff.start_standoff(
        crisis_context=str(crisis_context),
        modules=modules,
        use_id_rag=True,
        use_adaptive_idrag=True,
    )
    if isinstance(standoff.logger, LiveAegisLogger):
        standoff.logger.save_transcript()

    print("=" * 80)
    print(f"Session complete. Log saved to logs/experiments/{standoff.session_id}.json")
    print(f"Transcript saved to logs/experiments/{standoff.session_id}_demo.txt")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
