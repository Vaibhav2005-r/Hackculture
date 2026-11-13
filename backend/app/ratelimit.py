"""Sliding-window rate limiter (in-memory, per client IP and bucket).

Good for a single instance; behind a load balancer the same interface would sit on Redis.
"""

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status


class RateLimiter:
    def __init__(self) -> None:
        self._hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, bucket: str, limit: int, window_s: float = 60.0) -> None:
        now = time.monotonic()
        with self._lock:
            q = self._hits[(bucket, key)]
            while q and now - q[0] > window_s:
                q.popleft()
            if len(q) >= limit:
                retry = int(window_s - (now - q[0])) + 1
                raise HTTPException(
                    status.HTTP_429_TOO_MANY_REQUESTS,
                    f"Too many requests, try again in {retry}s",
                    headers={"Retry-After": str(retry)},
                )
            q.append(now)


    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


limiter = RateLimiter()


def client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def limit(bucket: str, per_minute: int):
    def dependency(request: Request) -> None:
        # RATE_LIMIT_MULTIPLIER lets an operator loosen every bucket at once (e.g. a booth where many
        # visitors share one IP) without touching code.
        from app.config import get_settings

        limiter.check(client_key(request), bucket, max(1, int(per_minute * get_settings().rate_limit_multiplier)))

    return dependency
