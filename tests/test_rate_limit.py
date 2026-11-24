import asyncio
import logging

from app.core.config import settings
from app.services.adapters.rate_limit import AdaptiveModelRateLimiter


class _Response:
    def __init__(self, headers):
        self.headers = headers
        self.status_code = 429


def _delay(headers, monkeypatch):
    monkeypatch.setattr(settings, "MODEL_RATE_LIMIT_JITTER_S", 0.0)
    monkeypatch.setattr(settings, "MODEL_RATE_LIMIT_BASE_COOLDOWN_S", 600.0)
    monkeypatch.setattr(settings, "MODEL_RATE_LIMIT_MAX_COOLDOWN_S", 1800.0)
    limiter = AdaptiveModelRateLimiter()
    return asyncio.run(
        limiter.register_rate_limit(
            _Response(headers),
            key="bedrock-test",
            logger=logging.getLogger("test-rate-limit"),
        )
    )


def test_short_retry_after_is_the_wait(monkeypatch):
    delay = _delay({"Retry-After": "2"}, monkeypatch)
    assert delay == 2.0


def test_missing_retry_after_uses_the_base_cooldown(monkeypatch):
    delay = _delay({}, monkeypatch)
    assert delay == 600.0


def test_long_retry_after_is_capped(monkeypatch):
    delay = _delay({"Retry-After": "99999"}, monkeypatch)
    assert delay == 1800.0
