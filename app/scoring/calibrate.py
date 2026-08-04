"""Platt / temperature calibration for reranker logits."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path("data/calibration/rerank.json")


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


class CalibrationStore:
    """Load fitted Platt/temperature parameters for policy-usable scores."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path or getattr(settings, "RERANK_CALIBRATION_PATH", DEFAULT_PATH))
        self.bias = 0.0
        self.temperature = 1.0
        self.version = "identity:missing"
        self.loaded = False
        self._load()

    def _load(self) -> None:
        try:
            if not self.path.exists():
                logger.info("Calibration artifact missing at %s; using identity", self.path)
                return
            data = json.loads(self.path.read_text())
            self.bias = float(data.get("b", data.get("bias", 0.0)))
            temp = float(data.get("T", data.get("temperature", 1.0)))
            self.temperature = temp if temp > 1e-6 else 1.0
            self.version = str(data.get("version", "platt:loaded"))
            self.loaded = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load calibration artifact: %s", exc)
            self.version = "identity:error"
            self.loaded = False

    def apply(self, logit: Optional[float], raw_score: Optional[float] = None) -> Tuple[float, str]:
        """Return calibrated p_hat and version tag."""
        if logit is None:
            if raw_score is None:
                return 0.0, self.version
            # Treat raw score as uncalibrated probability; convert to logit then calibrate.
            p = clamp(raw_score, 1e-6, 1.0 - 1e-6)
            logit = math.log(p / (1.0 - p))
        if not self.loaded:
            p_hat = 1.0 / (1.0 + math.exp(-float(logit)))
            return clamp(p_hat), self.version
        scaled = (float(logit) - self.bias) / self.temperature
        return clamp(1.0 / (1.0 + math.exp(-scaled))), self.version


_STORE: Optional[CalibrationStore] = None


def get_calibration_store() -> CalibrationStore:
    global _STORE
    if _STORE is None:
        _STORE = CalibrationStore()
    return _STORE


def apply_calibration(
    results: List[Dict[str, Any]],
    store: Optional[CalibrationStore] = None,
) -> List[Dict[str, Any]]:
    """Apply calibration to reranker result dicts in-place."""
    store = store or get_calibration_store()
    calibrated = []
    for item in results:
        logit = item.get("logit")
        raw = item.get("score", item.get("p_hat"))
        p_hat, version = store.apply(logit if logit is not None else None, raw_score=raw)
        updated = dict(item)
        updated["p_hat"] = p_hat
        updated["calibration_version"] = version
        calibrated.append(updated)
    return calibrated
