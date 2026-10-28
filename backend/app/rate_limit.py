"""In-memory sliding-window rate limit for the public demo.

State lives in one process, so on Lambda each warm instance counts separately — good enough to stop one
client from looping /ask; the OpenRouter key's spend limit is the real cap.
"""
import time
from collections import defaultdict, deque
from threading import Lock
from typing import Deque, Dict, Optional


class RateLimiter:
    def __init__(self, limit: int, window_s: int):
        self.limit = limit
        self.window_s = window_s
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, client_id: str, now: Optional[float] = None) -> bool:
        """Record a request; False once the client used `limit` requests in the last `window_s` seconds."""
        if self.limit <= 0:
            return True
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits[client_id]
            while hits and hits[0] <= now - self.window_s:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True


def client_id(forwarded_for: Optional[str], peer: Optional[str]) -> str:
    """Behind a Lambda Function URL every request comes from the local Web Adapter, so the peer address is
    useless. AWS appends the real source IP as the *last* X-Forwarded-For entry; earlier ones are client-supplied
    and spoofable."""
    if forwarded_for:
        return forwarded_for.split(",")[-1].strip()
    return peer or "unknown"
