import asyncio
import logging

import httpx

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


class _Client:
    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = 0

    async def post(self, url, json, headers):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        response = _Response({})
        response.status_code = 200
        return response


def test_a_read_timeout_does_not_send_a_second_generation(monkeypatch):
    monkeypatch.setattr(settings, "MODEL_RATE_LIMIT_MAX_ATTEMPTS", 6)
    limiter = AdaptiveModelRateLimiter()
    client = _Client([httpx.ReadTimeout("still running")])
    try:
        asyncio.run(
            limiter.post(
                client,
                "https://bedrock.example/model",
                key="gen",
                logger=logging.getLogger("test-rate-limit"),
                json={"messages": []},
                headers={},
            )
        )
    except httpx.ReadTimeout:
        pass
    else:
        raise AssertionError("the read timeout must stop the request")
    assert client.calls == 1


def test_a_write_timeout_does_not_send_a_second_generation(monkeypatch):
    monkeypatch.setattr(settings, "MODEL_RATE_LIMIT_MAX_ATTEMPTS", 6)
    limiter = AdaptiveModelRateLimiter()
    client = _Client([httpx.WriteTimeout("bytes already sent")])
    try:
        asyncio.run(
            limiter.post(
                client,
                "https://bedrock.example/model",
                key="gen",
                logger=logging.getLogger("test-rate-limit"),
                json={"messages": []},
                headers={},
            )
        )
    except httpx.WriteTimeout:
        pass
    else:
        raise AssertionError("the write timeout must stop the request")
    assert client.calls == 1


def test_a_refused_connection_retries_once(monkeypatch):
    monkeypatch.setattr(settings, "MODEL_RATE_LIMIT_MAX_ATTEMPTS", 6)
    limiter = AdaptiveModelRateLimiter()
    client = _Client([httpx.ConnectError("refused")])
    response = asyncio.run(
        limiter.post(
            client,
            "http://127.0.0.1:9/v1/chat/completions",
            key="local-gen",
            logger=logging.getLogger("test-rate-limit"),
            json={"messages": []},
            headers={},
        )
    )
    assert response.status_code == 200
    assert client.calls == 2


class _StatusClient:
    def __init__(self, rows):
        self.rows = list(rows)
        self.calls = 0

    async def post(self, url, json, headers):
        self.calls += 1
        status, response_headers = self.rows.pop(0)
        response = _Response(response_headers)
        response.status_code = status
        return response


def test_a_generation_server_error_does_not_send_a_second_request():
    limiter = AdaptiveModelRateLimiter()
    client = _StatusClient([(500, {})])
    response = asyncio.run(
        limiter.post(
            client,
            "https://generator.example/v1/chat/completions",
            key="gen-500",
            logger=logging.getLogger("test-rate-limit"),
            json={"messages": []},
            headers={},
            replay_safe=False,
        )
    )
    assert response.status_code == 500
    assert client.calls == 1


def test_a_generation_admission_retries_once_when_retry_after_is_short():
    limiter = AdaptiveModelRateLimiter()
    client = _StatusClient([(503, {"Retry-After": "0"}), (200, {})])
    response = asyncio.run(
        limiter.post(
            client,
            "http://127.0.0.1:9/v1/chat/completions",
            key="gen-503",
            logger=logging.getLogger("test-rate-limit"),
            json={"messages": []},
            headers={},
            replay_safe=False,
        )
    )
    assert response.status_code == 200
    assert client.calls == 2
