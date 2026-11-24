"""Readiness decisions that do not open a database."""

from __future__ import annotations


def cloud_health_status(*, index_exists: bool, provenance_valid: bool) -> str:
    """Return degraded when an existing index does not match the active embedder."""
    if index_exists and not provenance_valid:
        return "degraded"
    return "healthy"
