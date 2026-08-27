"""Atomic answer claims. Not the MAC ledger. Cap 8."""

from __future__ import annotations

import re
from typing import Iterable

CLAIM_CAP = 8
ABSTAIN_CLAIM_CAP = 3
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_CONJUNCT = re.compile(r"\s+\band\b\s+", re.IGNORECASE)


def split_claims(text: str, *, cap: int = CLAIM_CAP) -> list[str]:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return []
    sentences = [part.strip() for part in _SENTENCE.split(cleaned) if part.strip()]
    claims: list[str] = []
    for sentence in sentences:
        parts = [part.strip(" ,;") for part in _CONJUNCT.split(sentence) if part.strip(" ,;")]
        if len(parts) >= 2 and all(len(part.split()) >= 4 for part in parts):
            claims.extend(parts)
        else:
            claims.append(sentence)
        if len(claims) >= cap:
            break
    return claims[:cap]


def attach_citations(claims: Iterable[str], cited_passages: list[dict]) -> list[dict]:
    items = []
    for index, claim in enumerate(claims, start=1):
        items.append(
            {
                "claim_id": index,
                "claim": claim,
                "snippets": [
                    {
                        "doc_id": item.get("doc_id"),
                        "chunk_id": item.get("chunk_id"),
                        "text": item.get("text") or item.get("content") or "",
                    }
                    for item in cited_passages
                ],
                "uncited": not cited_passages,
            }
        )
    return items
