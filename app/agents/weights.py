"""Beta reliability priors → LCB agent weights."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path("data/calibration/agents.json")


def beta_lcb(a: float, b: float, delta: float = 0.05) -> float:
    """Conservative lower confidence bound for Beta(a,b) mean via quantile approx."""
    mean = a / max(a + b, 1e-9)
    # Normal approximation to Beta quantile for small enterprise prior tables.
    var = (a * b) / (max((a + b) ** 2 * (a + b + 1), 1e-9))
    z = 1.64485 if delta <= 0.05 else 1.28155
    return max(0.0, min(1.0, mean - z * (var ** 0.5)))


class ReliabilityWeights:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or getattr(settings, "AGENT_RELIABILITY_PATH", DEFAULT_PATH))
        self.priors: Dict[str, Dict[str, float]] = {
            "emr": {"a": 8, "b": 2},
            "guideline": {"a": 8, "b": 2},
            "safety": {"a": 10, "b": 2},
            "quality": {"a": 6, "b": 2},
            "critic": {"a": 7, "b": 2},
            "synthesis": {"a": 8, "b": 2},
            "retrieval": {"a": 9, "b": 2},
        }
        self._load()

    def _load(self) -> None:
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text())
                for key, value in data.items():
                    if isinstance(value, dict) and "a" in value and "b" in value:
                        self.priors[key] = {"a": float(value["a"]), "b": float(value["b"])}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Agent reliability load failed: %s", exc)

    def weights(self) -> Dict[str, float]:
        lcbs = {agent: beta_lcb(p["a"], p["b"]) for agent, p in self.priors.items()}
        total = sum(lcbs.values()) or 1.0
        return {agent: value / total for agent, value in lcbs.items()}
