from __future__ import annotations

import re
import threading
import time


class RateLimiter:
    """Enforce minimum spacing between API calls (thread-safe)."""

    def __init__(self, min_interval_sec: float) -> None:
        self.min_interval = max(0.0, min_interval_sec)
        self._lock = threading.Lock()
        self._last_request = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            elapsed = time.time() - self._last_request
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last_request = time.time()


def retry_delay_seconds(exc: Exception, attempt: int) -> float:
    """Use API-suggested delay for 429; exponential backoff otherwise."""
    msg = str(exc)
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        match = re.search(r"retry in ([\d.]+)s", msg, re.IGNORECASE)
        if match:
            return float(match.group(1)) + 2.0
        return 55.0
    return float(2**attempt)
