"""
Deterministic claim segmentation utilities.
"""

from __future__ import annotations

from typing import Dict, List
import re


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [p.strip() for p in parts if p and p.strip()]


def extract_claims(text: str) -> List[Dict[str, str]]:
    """Split text into deterministic claim units with stable IDs."""
    claims: List[Dict[str, str]] = []
    for idx, sentence in enumerate(_split_sentences(text), start=1):
        claim_id = f"claim_{idx:02d}"
        claims.append({"claim_id": claim_id, "claim_text": sentence})
    return claims
