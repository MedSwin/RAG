"""Shared adaptive rate limiting for remote model adapters."""

import asyncio
import email.utils
import logging
import random
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

from app.core.config import settings


def _header_seconds(value: Optional[str]) -> Optional[float]:
    """Parse retry/reset headers that may be seconds, HTTP dates, or compact durations."""
    if not value:
        return None

    raw = value.strip()
    try:
        seconds = float(raw)
        if seconds > 1_000_000_000:
            return max(0.0, seconds - time.time())
        return max(0.0, seconds)
    except ValueError:
        pass

    unit = raw[-2:].lower()
    number = raw[:-2]
    if unit == "ms":
        try:
            return max(0.0, float(number) / 1000.0)
        except ValueError:
            return None

    suffix = raw[-1:].lower()
    if suffix in {"s", "m"}:
        try:
            scale = 60.0 if suffix == "m" else 1.0
            return max(0.0, float(raw[:-1]) * scale)
        except ValueError:
            return None

    try:
        parsed = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, (parsed - datetime.now(timezone.utc)).total_seconds())


def _retry_after_seconds(response: httpx.Response) -> Optional[float]:
    headers = getattr(response, "headers", {})
    candidates = [
        headers.get("retry-after-ms"),
        headers.get("Retry-After-Ms"),
        headers.get("retry-after"),
        headers.get("Retry-After"),
        headers.get("x-ratelimit-reset-requests"),
        headers.get("x-ratelimit-reset-tokens"),
        headers.get("x-ratelimit-reset"),
    ]
    for value in candidates:
        seconds = _header_seconds(value)
        if seconds is not None:
            return seconds
    return None


def _remaining_quota(response: httpx.Response) -> Optional[int]:
    headers = getattr(response, "headers", {})
    for header in (
        "x-ratelimit-remaining-requests",
        "x-ratelimit-remaining-tokens",
    ):
        value = headers.get(header)
        if value is None:
            continue
        try:
            return int(value)
        except ValueError:
            continue
    return None


class AdaptiveModelRateLimiter:
    """Coordinates cooldowns across clients that target the same model deployment."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(settings.MODEL_RATE_LIMIT_MAX_CONCURRENCY)
        self._cooldown_until = 0.0
        self._consecutive_rate_limits = 0
        self._rate_limit_events = 0
        self._retry_events = 0
        self._last_delay = 0.0
        self._last_remaining_quota: Optional[int] = None

    async def wait_for_cooldown(self) -> None:
        while True:
            async with self._lock:
                delay = self._cooldown_until - time.monotonic()
            if delay <= 0:
                return
            await asyncio.sleep(delay)

    async def register_rate_limit(
        self,
        response: httpx.Response,
        *,
        key: str,
        logger: logging.Logger,
    ) -> float:
        async with self._lock:
            self._consecutive_rate_limits += 1
            self._rate_limit_events += 1
            header_delay = _retry_after_seconds(response)
            fallback_delay = min(
                settings.MODEL_RATE_LIMIT_MAX_COOLDOWN_S,
                settings.MODEL_RATE_LIMIT_BASE_COOLDOWN_S * (2 ** (self._consecutive_rate_limits - 1)),
            )
            jitter = random.uniform(0, settings.MODEL_RATE_LIMIT_JITTER_S)
            # Root Cause vs Logic: provider retry headers can advertise long
            # cooldowns that are still shorter than the full rate-limit window.
            # The logic now respects the configured maximum cooldown instead of
            # truncating to a few seconds, so large ingest runs can actually
            # clear the quota boundary before retrying.
            bounded_header_delay = (
                min(header_delay, settings.MODEL_RATE_LIMIT_MAX_COOLDOWN_S)
                if header_delay is not None
                else None
            )
            # Root Cause vs Logic: some cloud embedding deployments emit short
            # retry hints even when repeated 429s show the real quota window is
            # longer. We therefore honor the provider hint but never wait less
            # than the exponential fallback, so repeated failures back off more
            # aggressively instead of cycling at the same interval.
            delay = max(bounded_header_delay if bounded_header_delay is not None else 0.0, fallback_delay) + jitter
            self._cooldown_until = max(self._cooldown_until, time.monotonic() + delay)
            self._last_delay = delay

        logger.warning("Model rate limit reached for %s; cooling down for %.2fs", key, delay)
        return delay

    async def register_success(self, response: httpx.Response) -> None:
        async with self._lock:
            self._consecutive_rate_limits = 0
            remaining = _remaining_quota(response)
            reset_delay = _retry_after_seconds(response)
            self._last_remaining_quota = remaining
            if remaining == 0 and reset_delay is not None:
                self._cooldown_until = max(self._cooldown_until, time.monotonic() + reset_delay)

    async def post(
        self,
        client: httpx.AsyncClient,
        url: str,
        *,
        key: str,
        logger: logging.Logger,
        json: Dict[str, Any],
        headers: Dict[str, str],
        fail_open_after_s: float | None = None,
    ) -> httpx.Response:
        last_error: Optional[BaseException] = None
        last_response: Optional[httpx.Response] = None

        for attempt in range(1, settings.MODEL_RATE_LIMIT_MAX_ATTEMPTS + 1):
            await self.wait_for_cooldown()

            try:
                async with self._semaphore:
                    response = await client.post(url, json=json, headers=headers)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                self._retry_events += 1
                if attempt == settings.MODEL_RATE_LIMIT_MAX_ATTEMPTS:
                    raise
                await asyncio.sleep(_retry_backoff_seconds(attempt))
                continue

            status_code = getattr(response, "status_code", 200)
            if status_code == 429:
                last_response = response
                delay = await self.register_rate_limit(response, key=key, logger=logger)
                if fail_open_after_s is not None and delay >= fail_open_after_s:
                    # Root Cause vs Logic: long provider cooldowns can block the
                    # entire benchmark for far longer than the remaining eval
                    # window. The reranker can safely fail open to the original
                    # candidate order, which preserves liveness while still
                    # recording the quota pressure in diagnostics.
                    raise RuntimeError(
                        f"Rate limit cooldown {delay:.2f}s exceeds fail-open threshold for {key}"
                    )
                continue

            if status_code in {500, 502, 503, 504}:
                last_response = response
                self._retry_events += 1
                if attempt == settings.MODEL_RATE_LIMIT_MAX_ATTEMPTS:
                    response.raise_for_status()
                await asyncio.sleep(_retry_backoff_seconds(attempt))
                continue

            await self.register_success(response)
            return response

        if last_response is not None:
            last_response.raise_for_status()
            return last_response
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Model request failed without response for {key}")


def _retry_backoff_seconds(attempt: int) -> float:
    delay = min(
        settings.MODEL_RATE_LIMIT_MAX_COOLDOWN_S,
        settings.MODEL_RATE_LIMIT_BASE_COOLDOWN_S * (2 ** (attempt - 1)),
    )
    return delay + random.uniform(0, settings.MODEL_RATE_LIMIT_JITTER_S)


_limiters: Dict[str, AdaptiveModelRateLimiter] = {}
_limiters_lock = asyncio.Lock()


async def request_with_model_rate_limit(
    client: httpx.AsyncClient,
    url: str,
    *,
    rate_limit_key: str,
    logger: logging.Logger,
    json: Dict[str, Any],
    headers: Dict[str, str],
    fail_open_after_s: float | None = None,
) -> httpx.Response:
    """POST through a shared per-model limiter.

    Root Cause vs Logic: 429s were retried as independent request failures, so
    parallel model clients kept sending new work during provider cooldowns. The
    limiter stores cooldown state by model deployment key, waits before new
    attempts, and lets provider retry/reset headers drive the next retry.
    """
    async with _limiters_lock:
        limiter = _limiters.setdefault(rate_limit_key, AdaptiveModelRateLimiter())
    return await limiter.post(
        client,
        url,
        key=rate_limit_key,
        logger=logger,
        json=json,
        headers=headers,
        fail_open_after_s=fail_open_after_s,
    )


def rate_limit_snapshot() -> Dict[str, Dict[str, Any]]:
    """Return a snapshot of shared model throttling state keyed by deployment."""
    snapshot: Dict[str, Dict[str, Any]] = {}
    for key, limiter in list(_limiters.items()):
        snapshot[key] = {
            "rate_limit_events": limiter._rate_limit_events,
            "retry_events": limiter._retry_events,
            "cooldown_until": limiter._cooldown_until,
            "last_delay": limiter._last_delay,
            "last_remaining_quota": limiter._last_remaining_quota,
            "consecutive_rate_limits": limiter._consecutive_rate_limits,
            "max_concurrency": settings.MODEL_RATE_LIMIT_MAX_CONCURRENCY,
        }
    return snapshot
