"""ANN label identity. A chunk id is unique only inside one organisation."""

from __future__ import annotations

from typing import Any, Optional, Tuple

LABEL_SEPARATOR = "\x1f"


def encode_label(chunk_id: str, org_id: Optional[str] = None) -> str:
    """Store the organisation with the chunk id when the organisation is known."""
    chunk = str(chunk_id or "").strip()
    org = str(org_id or "").strip()
    if not org:
        return chunk
    return f"{org}{LABEL_SEPARATOR}{chunk}"


def decode_label(value: str) -> Tuple[Optional[str], str]:
    """Return the organisation and the chunk id. A legacy label has no organisation."""
    text = str(value or "")
    if LABEL_SEPARATOR not in text:
        return None, text
    org, chunk = text.split(LABEL_SEPARATOR, 1)
    return (org or None), chunk


def resolve_label(mapping: Any, label: Any) -> Tuple[Optional[str], Optional[str]]:
    """Read one ANN label. Skip a blank mapping entry."""
    resolver = getattr(mapping, "resolve", None)
    if resolver is not None:
        org_id, chunk_id = resolver(label)
        if not chunk_id:
            return None, None
        return org_id, chunk_id
    raw = mapping.get(str(int(label)))
    if not raw:
        return None, None
    org_id, chunk_id = decode_label(str(raw))
    if not chunk_id:
        return None, None
    return org_id, chunk_id
